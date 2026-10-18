import math

import numpy as np
import torch

import igl
import pytest
import trimesh

import genesis as gs
from genesis.utils.element import create_tetrahedral_grid
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_analytic_container_collision(n_envs, show_viewer):
    # Binary-exact contact coordinates and timestep preserve velocity through finite differencing.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 1024.0,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.04,
            lower_bound=(-0.2, -0.2, -0.2),
            upper_bound=(1.4, 1.6, 1.4),
            boundary_pitcher_shell=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.125, 0.125),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -4.0, 3.0),
            camera_lookat=(0.5, 0.5, 0.5),
        ),
        show_viewer=show_viewer,
    )
    positions = np.array(
        [
            (1.03125, 0.0, 0.5),
            (1.09375, 0.0, 0.5),
            (0.0, 0.0, -0.03125),
            (0.0, 0.0, -0.09375),
            (1.03125, 0.0, -0.03125),
            (0.0, 0.0, 1.2),
        ]
    )
    velocities = np.array(
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, -1.0),
            (0.0, 0.0, 0.0),
        ]
    )
    expected = np.array(
        [
            (1.0, 0.0, 0.5),
            (1.125, 0.0, 0.5),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -0.125),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.2),
        ]
    )
    liquids = []
    for pos in positions:
        liquids.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=pos,
                    size=(0.04, 0.04, 0.04),
                ),
                material=gs.materials.PBD.Liquid(sampler="regular"),
            )
        )
    scene.build(n_envs=n_envs)
    for is_tilted in (False, True):
        origin = (0.0, 0.25, 0.0) if is_tilted else (0.0, 0.0, 0.0)
        scene.pbd_solver.set_pitcher_pose(origin, axis=(0.0, 2.0, 0.0) if is_tilted else (0.0, 0.0, 1.0))
        coordinates = (0, 2, 1) if is_tilted else (0, 1, 2)
        for entity, pos, vel in zip(liquids, positions[:, coordinates] + origin, velocities[:, coordinates]):
            entity.set_particles_pos(pos)
            entity.set_particles_vel(vel)
        scene.step()
        for entity, pos in zip(liquids, expected[:, coordinates] + origin):
            assert_allclose(entity.get_particles_pos(), pos, atol=2e-6)
            assert_allclose(entity.get_particles_vel(), 0.0, atol=2e-5)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("contact_distances", [(0.0, 0.0), (5e-5, 1.5e-4)])
def test_unified_elastic_projection(n_envs, contact_distances, show_viewer, asset_tmp_path):
    thickness, detection_distance = contact_distances
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=0.2,
            max_solver_iterations=1,
            constraint_acceleration=0.9,
            collision_thickness=thickness,
            contact_detection_distance=detection_distance,
            is_recording_constraint_history=True,
            is_recording_edge_contacts=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, -4.0, 2.5),
            camera_lookat=(2.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    entities = []
    for i_entity, (size, stretch_relaxation, volume_relaxation, volume_compliance) in enumerate(
        ((0.03, 0.0, 0.5, 0.0), (0.1, 0.0, 1.0, 0.0), (0.3, 0.15, 0.25, 0.0), (0.1, 0.0, 0.5, 1e-8))
    ):
        entities.append(
            scene.add_entity(
                morph=gs.morphs.TetrahedralMesh(
                    pos=(0.4 * i_entity, 0.0, 0.5),
                    vertices=np.array(((0.0, 0.0, 0.0), (size, 0.0, 0.0), (0.0, size, 0.0), (0.0, 0.0, size))),
                    elements=((0, 1, 2, 3),),
                ),
                material=gs.materials.PBD.Elastic(
                    volume_compliance=volume_compliance,
                    stretch_relaxation=stretch_relaxation,
                    volume_relaxation=volume_relaxation,
                ),
            )
        )
    cloth_path = asset_tmp_path / "unified_hinge.obj"
    trimesh.Trimesh(
        vertices=((-0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, np.sqrt(0.03), 0.0), (0.0, -np.sqrt(0.03), 0.0)),
        faces=((0, 1, 2), (1, 0, 3)),
        process=False,
    ).export(cloth_path)
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            pos=(1.6, 0.0, 0.5),
            file=cloth_path,
        ),
        material=gs.materials.PBD.Cloth(
            stretch_compliance=0.0,
            bending_compliance=0.0,
            stretch_relaxation=0.1,
            bending_relaxation=0.2,
            air_resistance=0.0,
        ),
    )
    contact_elastic = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(2.0, 0.0, 0.5),
            vertices=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
            elements=((0, 1, 2, 3),),
        ),
        material=gs.materials.PBD.Elastic(
            stretch_relaxation=0.0,
            volume_relaxation=0.5,
        ),
    )
    contact_rigid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(2.04, 0.04, 0.5875),
            size=(0.3, 0.3, 0.015),
        ),
        material=gs.materials.Rigid(
            is_coup_reaction_enabled=False,
        ),
    )
    probe_path = asset_tmp_path / "unified_contact_probe.obj"
    trimesh.Trimesh(
        vertices=((0.005, 0.03, -0.005), (0.003, 0.028, 0.08), (0.007, 0.028, 0.08), (0.005, 0.032, 0.08)),
        faces=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        process=False,
    ).export(probe_path)
    face_entities = []
    probes = []
    for i_contact in range(2):
        vertices = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, -0.1)]
        elements = ((0, 2, 1, 3),)
        if i_contact == 0:
            vertices.append((0.02, 0.03, -0.006))
            elements = ((0, 2, 1, 4), (0, 1, 3, 4), (0, 3, 2, 4), (1, 2, 3, 4))
        face_entities.append(
            scene.add_entity(
                morph=gs.morphs.TetrahedralMesh(
                    pos=(2.5 + 0.5 * i_contact, 0.0, 0.5),
                    vertices=vertices,
                    elements=elements,
                ),
                material=gs.materials.PBD.Elastic(
                    stretch_relaxation=0.0,
                    volume_relaxation=0.0,
                ),
            )
        )
        probes.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    pos=(2.515 if i_contact == 0 else 3.0, 0.0, 0.5 if i_contact == 0 else 0.51),
                    file=probe_path,
                ),
                material=gs.materials.Rigid(
                    is_coup_reaction_enabled=False,
                ),
            )
        )
    mixed_contact_rigid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(3.05, 0.05, 0.4975),
            size=(0.5, 0.5, 0.007),
        ),
        material=gs.materials.Rigid(
            is_coup_reaction_enabled=False,
        ),
    )
    edge_elastic = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(4.0, 0.0, 0.5),
            vertices=0.05 * np.array(((-1.0, -1.0, 0.0), (2.0, 2.0, 0.0), (-1.0, 2.0, 0.0), (-0.7, 1.0, -0.2))),
            elements=((0, 2, 1, 3),),
        ),
        material=gs.materials.PBD.Elastic(
            stretch_relaxation=0.0,
            volume_relaxation=0.0,
        ),
    )
    edge_probe_path = asset_tmp_path / "unified_edge_contact_probe.obj"
    trimesh.Trimesh(
        vertices=0.05 * np.array(((-0.5, 0.0, -1.0), (-0.5, 0.0, 2.0), (2.5, 0.0, 2.0), (0.5, -0.2, 0.5))),
        faces=((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
        process=False,
    ).export(edge_probe_path)
    edge_rigids = []
    for i_contact in range(2):
        edge_rigids.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    pos=(4.0, 0.0, 1.0 + 0.5 * i_contact),
                    file=edge_probe_path,
                    convexify=False,
                ),
                material=gs.materials.Rigid(
                    is_coup_reaction_enabled=False,
                ),
            )
        )
    proximity_rigids = []
    if thickness > 0.0:
        proximity_path = asset_tmp_path / "unified_edge_proximity.obj"
        trimesh.Trimesh(
            vertices=((0.0, -0.02, 0.0), (0.0, 0.02, 0.0), (0.02, 0.0, -0.02), (-0.02, 0.0, -0.02)),
            faces=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
            process=False,
        ).export(proximity_path)
        for z in (0.5, 1.5):
            proximity_rigids.append(
                scene.add_entity(
                    morph=gs.morphs.Mesh(
                        pos=(5.0, 0.0, z),
                        file=proximity_path,
                        convexify=False,
                    ),
                    material=gs.materials.Rigid(
                        is_coup_reaction_enabled=False,
                    ),
                )
            )
    scene.build(n_envs=n_envs)
    if thickness > 0.0:
        for i_rigid, rigid in enumerate((*probes, mixed_contact_rigid)):
            rigid.set_pos((10.0 + i_rigid, 0.0, 1.0))
        contact_elastic.fix_particles([0, 1, 2])
        for gap in (0.5 * thickness, 1.5 * thickness, 1.1 * detection_distance):
            positions = contact_elastic.init_particles.copy()
            contact_elastic.set_particles_pos(positions)
            contact_rigid.set_pos((2.04, 0.04, 0.6075 + gap))
            scene.pbd_solver.project_collision()
            positions[3, 2] -= max(thickness - gap, 0.0)
            assert_allclose(contact_elastic.get_particles_pos(), positions, atol=2e-7)
        contact_rigid.set_pos((14.0, 0.0, 1.0))
        face_entity = face_entities[0]
        face_entity.fix_particles([3, 4])
        gap = 0.5 * thickness
        probes[0].set_pos((2.515, 0.0, 0.505 + gap))
        positions = face_entity.init_particles.copy()
        face_entity.set_particles_pos(positions)
        scene.pbd_solver.project_collision()
        barycentric = np.array((0.5, 0.2, 0.3))
        positions[:3, 2] -= (thickness - gap) * barycentric / (barycentric**2).sum()
        assert_allclose(face_entity.get_particles_pos(), positions, atol=2e-7)
        probes[0].set_pos((10.0, 0.0, 1.0))
        for gap in (-0.5 * thickness, 0.0, 0.5 * thickness, 1.5 * thickness, 1.1 * detection_distance):
            positions = np.array(((-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.02), (0.0, -0.02, 0.02)))
            positions += (5.0, 0.0, 0.5 + gap)
            edge_elastic.set_particles_pos(positions)
            scene.pbd_solver.project_collision()
            positions[:2, 2] += max(thickness - gap, 0.0)
            assert_allclose(edge_elastic.get_particles_pos(), positions, atol=2e-7)
            assert_allclose(proximity_rigids[0].get_pos(), (5.0, 0.0, 0.5), atol=2e-7)
        for offset, count in ((thickness, 2), (detection_distance, 1)):
            positions = np.array(((-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.02), (0.0, -0.02, 0.02)))
            positions += (5.0, 0.0, 0.5 + 0.5 * thickness)
            edge_elastic.set_particles_pos(positions)
            proximity_rigids[1].set_pos((5.005, 0.0, 0.5 - offset))
            scene.pbd_solver.project_collision()
            # The separated pair contributes to averaging only while it remains inside the detection band.
            positions[:2, 2] += 0.5 * thickness / count
            assert_allclose(edge_elastic.get_particles_pos(), positions, atol=2e-7)
        positions = contact_elastic.init_particles.copy()
        positions[3, 2] = 0.5 * thickness
        contact_elastic.set_particles_pos(positions)
        scene.pbd_solver.project_collision()
        positions[3, 2] = thickness
        assert_allclose(contact_elastic.get_particles_pos(), positions, atol=2e-7)
        return
    initial_state = scene.get_state()
    contact_positions = contact_elastic.init_particles.copy()
    contact_positions[3, 2] -= 0.02
    contact_elastic.set_particles_pos(contact_positions)
    contact_elastic.fix_particles([0, 1, 2])
    contact_positions[3, 2] = tensor_to_array(contact_rigid.get_verts())[..., 2].max()
    for i_contact, (entity, probe) in enumerate(zip(face_entities, probes)):
        entity.fix_particles([3, 4] if i_contact == 0 else [0, 3])
        velocity = 0.0 if i_contact == 0 else -2.5
        probe.set_dofs_velocity((0.0, 0.0, velocity, 0.0, 0.0, 0.0))
    expected_positions = []
    for entity in entities:
        rest = entity.init_particles
        positions = rest.copy()
        positions[3, 2] = rest[0, 2] + 0.8 * (rest[3, 2] - rest[0, 2])
        entity.set_particles_pos(positions)
        entity.fix_particles([0, 1, 2])
        inverse_mass = entity.n_particles / tensor_to_array(entity.get_mass()).mean()
        material = entity.material
        edge = positions[3] - positions[:3]
        lengths = np.linalg.norm(edge, axis=-1)
        lengths_rest = np.linalg.norm(rest[3] - rest[:3], axis=-1)
        stretch_delta = (
            -material.stretch_relaxation
            * (lengths - lengths_rest)[:, None]
            * edge
            / lengths[:, None]
            * inverse_mass
            / (inverse_mass + material.stretch_compliance / scene.dt**2)
        ).sum(axis=0)
        base_area_gradient = np.cross(rest[1] - rest[0], rest[2] - rest[0]) / 6.0
        volume_residual = np.dot(positions[3] - rest[3], base_area_gradient)
        volume_delta = (
            -material.volume_relaxation
            * volume_residual
            * inverse_mass
            * base_area_gradient
            / (inverse_mass * np.dot(base_area_gradient, base_area_gradient) + material.volume_compliance / scene.dt**2)
        )
        positions[3] += stretch_delta + volume_delta
        expected_positions.append(positions)
    assert cloth.n_particles == 4
    cloth_positions = cloth.init_particles.copy()
    free_idx = np.argmin(cloth_positions[:, 1])
    fixed_idx = np.flatnonzero(np.arange(cloth.n_particles) != free_idx)
    cloth_positions[free_idx, 2] += 0.05
    cloth.set_particles_pos(cloth_positions)
    cloth.fix_particles(fixed_idx)
    hinge_idx = np.flatnonzero(np.isclose(cloth_positions[:, 1], 0.0))
    edges = cloth_positions[free_idx] - cloth_positions[hinge_idx]
    lengths = np.linalg.norm(edges, axis=-1)
    rest_lengths = np.linalg.norm(cloth.init_particles[free_idx] - cloth.init_particles[hinge_idx], axis=-1)
    stretch_delta = (
        -cloth.material.stretch_relaxation * (lengths - rest_lengths)[:, None] * edges / lengths[:, None]
    ).sum(axis=0)
    y = cloth_positions[free_idx, 1]
    z = cloth_positions[free_idx, 2] - cloth.init_particles[free_idx, 2]
    angle = np.arctan2(z, -y)
    bending_delta = cloth.material.bending_relaxation * angle * np.array((0.0, -z, y))
    cloth_positions[free_idx] += stretch_delta + bending_delta
    scene.step()
    assert_allclose(contact_elastic.get_particles_pos(), contact_positions, atol=2e-7)
    for i_contact, (entity, probe) in enumerate(zip(face_entities, probes)):
        positions = entity.init_particles.copy()
        vertices = tensor_to_array(probe.get_verts()).reshape((-1, probe.n_verts, 3))[0]
        point = vertices[vertices[:, 2].argmin()]
        barycentric = np.zeros(3)
        barycentric[1] = (point[0] - positions[0, 0]) / (positions[1, 0] - positions[0, 0])
        barycentric[2] = (point[1] - positions[0, 1]) / (positions[2, 1] - positions[0, 1])
        barycentric[0] = 1.0 - barycentric[1:].sum()
        weights = np.array((1.0 if i_contact == 0 else 0.0, 1.0, 1.0))
        residual = point[2] - positions[0, 2]
        positions[:3, 2] += residual * weights * barycentric / (weights * barycentric**2).sum()
        if i_contact == 1:
            upper = tensor_to_array(mixed_contact_rigid.get_verts())[..., 2].max()
            positions[:3, 2] += weights * (upper - entity.init_particles[0, 2])
        assert_allclose(entity.get_particles_pos(), positions, atol=2e-7)
    assert_allclose(cloth.get_particles_pos(), cloth_positions, atol=2e-7)
    for entity, expected in zip(entities, expected_positions):
        positions = tensor_to_array(entity.get_particles_pos()).reshape((-1, entity.n_particles, 3))
        assert_allclose(positions, expected[None], atol=2e-7)
    history = scene.pbd_solver.get_constraint_history()
    elastic_elems_idx = [entity.elem_start for entity in (*entities, contact_elastic)]
    assert (history.volume[0, 1, :, elastic_elems_idx].abs() <= history.volume[0, 0, :, elastic_elems_idx].abs()).all()
    assert (history.bending[0, 1].abs() < history.bending[0, 0].abs()).all()
    entities.append(cloth)
    scene.reset(initial_state)
    for i_rigid, rigid in enumerate((contact_rigid, *probes, mixed_contact_rigid)):
        rigid.set_pos((10.0 + i_rigid, 0.0, 1.0))
    positions = tensor_to_array(edge_elastic.get_particles_pos(), dtype=np.float64).reshape((-1, 4, 3))
    rigid_faces = edge_rigids[0].links[0].geoms[0].init_faces.astype(np.int64)
    surface_faces = edge_elastic.surface_triangles.astype(np.int64)
    edge_displacements = []
    for i_contact, edge_rigid in enumerate(edge_rigids):
        edge_elastic.set_particles_pos(positions)
        edge_elastic.set_particles_vel(0.0)
        edge_rigid.set_pos((4.0 + 0.005 * i_contact, 0.0, 0.5))
        rigid_positions = tensor_to_array(edge_rigid.get_verts(), dtype=np.float64).reshape((-1, 4, 3))
        assert (np.abs(igl.winding_number(rigid_positions[0], rigid_faces, positions[0])) < 0.5).all()
        assert (np.abs(igl.winding_number(positions[0], surface_faces, rigid_positions[0])) < 0.5).all()
        diagnostics = scene.pbd_solver.get_collision_diagnostics()
        assert_equal(diagnostics.triangle_intersections, 4 * (i_contact + 1))
        assert_equal(diagnostics.edge_contacts, i_contact + 1)
        assert_allclose(diagnostics.edge_penetration, 0.025 / math.sqrt(2.0), atol=2e-7)
        if i_contact == 0:
            edge_elastic.fix_particles([0, 1])
            scene.step()
            assert_allclose(edge_elastic.get_particles_pos().reshape((-1, 4, 3)), positions, atol=2e-7)
            edge_elastic.release_particle()
        scene.step()
        assert_allclose(edge_rigid.get_verts().reshape((-1, 4, 3)), rigid_positions, atol=2e-7)
        # Projecting (x, x) onto the vertical rigid edge at (rigid_x, 0) gives x=rigid_x/2.
        rigid_x = -0.025 + 0.005 * i_contact
        parameter = (rigid_x / 2.0 + 0.05) / 0.15
        interpolation = np.array((1.0 - parameter, parameter))
        correction = np.zeros((4, 3))
        correction[:2, :2] = rigid_x / 2.0 * interpolation[:, None] * np.array((1.0, -1.0)) / (interpolation**2).sum()
        edge_displacements.append(correction)
        expected_displacements = np.mean(edge_displacements, axis=0)
        diagnostics_after = scene.pbd_solver.get_collision_diagnostics()
        corrected = positions[0] + expected_displacements
        direction = corrected[1, :2] - corrected[0, :2]
        offset = np.array((3.975, 0.0)) - corrected[0, :2]
        expected_gap = max(0.0, (direction[0] * offset[1] - direction[1] * offset[0]) / np.linalg.norm(direction))
        assert_allclose(diagnostics_after.edge_penetration, expected_gap, atol=2e-7)
        displacements = tensor_to_array(edge_elastic.get_particles_pos()).reshape((-1, 4, 3)) - positions
        assert_allclose(displacements, expected_displacements, atol=2e-7)
        scene.pbd_solver.project_elastic_constraints()
        assert_allclose(
            edge_elastic.get_particles_pos().reshape((-1, 4, 3)), positions + expected_displacements, atol=2e-7
        )
    edge_rigids[1].set_pos((4.0, 0.0, 1.5))
    positions = edge_elastic.init_particles.copy()
    positions[:, 0] -= 0.024
    edge_elastic.set_particles_pos(positions)
    edge_elastic.set_particles_vel(0.0)
    mixed_contact_rigid.set_pos((4.0, -0.298, 0.5))
    assert_equal(scene.pbd_solver.get_collision_diagnostics().edge_contacts, 1)
    scene.step()
    # Pushing the lower endpoint to the box face separates the crossing edges before their projection.
    positions[0, 1] = tensor_to_array(mixed_contact_rigid.get_verts())[..., 1].max()
    assert_allclose(edge_elastic.get_particles_pos(), positions, atol=2e-7)
    assert_equal(scene.pbd_solver.get_collision_diagnostics().triangle_intersections, 0)
    mixed_contact_rigid.set_pos((13.0, 0.0, 1.0))
    positions = 0.05 * np.array(((0.5, -1.5, 0.2), (-0.9, 0.8, 1.5), (0.1, 0.5, -0.3), (0.3, -0.1, 0.1)))
    positions += (4.0, 0.0, 0.5)
    edge_elastic.set_particles_pos(positions)
    edge_elastic.set_particles_vel(0.0)
    edge_elastic.fix_particles([0, 1, 2])
    diagnostics = scene.pbd_solver.get_collision_diagnostics()
    assert (diagnostics.triangle_intersections > 0).all()
    assert_equal(diagnostics.edge_contacts, 0)
    scene.step()
    # The closest rigid face has normal proportional to (2, -5, -2); crossing edges lie on its admissible side.
    positions[3] += 0.005 / 33.0 * np.array((2.0, -5.0, -2.0))
    assert_allclose(edge_elastic.get_particles_pos(), positions, atol=2e-7)
    edge_elastic.release_particle()
    positions = tensor_to_array(edge_elastic.init_particles, dtype=np.float64)
    edge_elastic.set_particles_pos(positions)
    edge_elastic.set_particles_vel(0.0)
    edge_elastic.fix_particles([2, 3])
    edge_rigids[1].set_pos((3.96, -0.04, 0.52))
    assert (scene.pbd_solver.get_collision_diagnostics().edge_contacts > 1).all()
    scene.step()
    history = scene.pbd_solver.get_edge_contact_history()
    assert_allclose(
        history.positions[0, 0, :, edge_elastic.particle_start : edge_elastic.particle_end], positions, atol=2e-7
    )
    assert (history.is_detected[0, 0].sum(dim=-1) > 1).all()
    # Vertex 0 receives three constraints; vertex 1 receives one. The other endpoints are fixed.
    expected_displacements = edge_displacements[0].copy()
    rigid_start = np.array((3.935, -0.04, 0.47))
    rigid_direction = np.array((0.05, -0.01, 0.075))
    for endpoint in (2, 3):
        elastic_direction = positions[endpoint] - positions[0]
        directions = np.column_stack((elastic_direction, -rigid_direction))
        parameter, rigid_parameter = np.linalg.lstsq(directions, rigid_start - positions[0], rcond=None)[0]
        closest = positions[0] + parameter * elastic_direction
        rigid_closest = rigid_start + rigid_parameter * rigid_direction
        expected_displacements[0] += (rigid_closest - closest) / (1.0 - parameter)
    expected_displacements[0] /= 3
    positions += expected_displacements
    assert_allclose(edge_elastic.get_particles_pos(), positions, atol=2e-7)
    assert_equal(scene.pbd_solver.get_collision_diagnostics().edge_contacts, 1)
    assert_equal(scene.pbd_solver.get_edge_contact_history().is_detected, history.is_detected)
    scene.pbd_solver.project_elastic_constraints()
    assert_allclose(edge_elastic.get_particles_pos(), positions, atol=2e-7)
    edge_elastic.release_particle()
    scene.reset(initial_state)
    for entity in entities:
        entity.fix_particles()
    scene.step()
    for entity in entities:
        assert_allclose(entity.get_particles_pos(), entity.init_particles, atol=2e-7)
        entity.release_particle()
        entity.set_particles_vel((0.02, -0.03, 0.04))
    scene.step()
    for entity in entities:
        assert_allclose(
            entity.get_particles_pos(), entity.init_particles + scene.dt * np.array((0.02, -0.03, 0.04)), atol=2e-7
        )
        assert_allclose(entity.get_particles_vel(), (0.02, -0.03, 0.04), atol=2e-5)

    positions_before_reset = [entity.get_particles_pos().clone() for entity in entities]
    scene.reset(initial_state, envs_idx=0 if n_envs else None)
    for entity, previous in zip(entities, positions_before_reset):
        positions = entity.get_particles_pos().reshape((-1, entity.n_particles, 3))
        assert_allclose(positions[0], entity.init_particles, atol=2e-7)
        if n_envs:
            assert_allclose(positions[1:], previous[1:], atol=2e-7)

    scene.reset(initial_state)
    momenta = []
    for entity in entities:
        positions = entity.init_particles
        relative = positions - positions.mean(axis=0)
        velocities = np.cross((-2.0, 1.0, 3.0), relative) + (0.02, -0.03, 0.04) + relative * (0.3, -0.2, 0.1)
        entity.set_particles_vel(velocities)
        momenta.append((velocities.mean(axis=0), np.cross(relative, velocities).mean(axis=0)))
    scene.step()
    for entity, (linear, angular) in zip(entities, momenta):
        positions = tensor_to_array(entity.get_particles_pos())
        velocities = tensor_to_array(entity.get_particles_vel())
        relative = positions - positions.mean(axis=-2, keepdims=True)
        assert_allclose(velocities.mean(axis=-2), linear, atol=2e-6)
        assert_allclose(np.cross(relative, velocities).mean(axis=-2), angular, atol=1e-7)

    scene.reset(initial_state)
    entity = next(
        entity
        for entity in entities
        if isinstance(entity.material, gs.materials.PBD.Elastic) and entity.material.volume_compliance > 0.0
    )
    positions = entity.init_particles.copy()
    height = positions[3, 2] - positions[0, 2]
    positions[3, 2] -= 0.2 * height
    entity.set_particles_pos(positions)
    entity.fix_particles([0, 1, 2])
    scene.step()
    for _ in range(2):
        scene.pbd_solver.project_elastic_constraints()
    recovered_height = entity.get_particles_pos()[..., 3, 2] - positions[0, 2]
    assert (recovered_height >= 0.97 * height).all()
    assert (recovered_height <= height).all()

    entity = entities[0]
    scene.reset(initial_state)
    recovering_positions = entity.init_particles.copy()
    height = recovering_positions[3, 2] - recovering_positions[0, 2]
    recovering_positions[3, 2] = recovering_positions[0, 2] - 2.0 * height
    entity.set_particles_pos(recovering_positions)
    entity.fix_particles([0, 1, 2])
    scene.step()
    assert_allclose(scene.pbd_solver.get_constraint_residuals().volume[..., 0], -1.5, atol=2e-5)
    scene.step()
    assert_allclose(entity.get_particles_pos(), entity.init_particles, atol=2e-7)
    scene.reset(initial_state)
    inverted_positions = entity.init_particles.copy()
    inverted_positions[3, 2] = 2.0 * inverted_positions[0, 2] - inverted_positions[3, 2]
    entity.set_particles_pos(inverted_positions)
    entity.fix_particles()
    with pytest.raises(gs.GenesisException, match="inverted tetrahedron"):
        scene.step()
    scene.reset(initial_state)
    invalid_state = scene.pbd_solver.get_state(0)
    invalid_state.vel[:] = np.nan
    scene.pbd_solver.set_state(0, invalid_state)
    with pytest.raises(gs.GenesisException, match="non-finite"):
        scene.step()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_elastic_damping(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbd_options=gs.options.PBDUnifiedOptions(
            max_solver_iterations=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -1.2, 1.1),
            camera_lookat=(0.05, -0.7, 0.55),
        ),
        show_viewer=show_viewer,
    )
    damping_entities = []
    # Binary-exact offsets within one exponent range give comparable single-precision position differences.
    for i_entity, damping in enumerate((0.0, 40.0, 4000.0)):
        damping_entities.append(
            scene.add_entity(
                morph=gs.morphs.TetrahedralMesh(
                    pos=(0.0, -0.625 - 0.125 * i_entity, 0.5),
                    vertices=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
                    elements=((0, 1, 2, 3),),
                ),
                material=gs.materials.PBD.Elastic(damping=damping),
            )
        )
    scene.build(n_envs=n_envs)
    initial_state = scene.get_state()
    for is_fixed in (False, True):
        scene.reset(initial_state)
        for entity in damping_entities:
            relative = entity.init_particles - entity.init_particles.mean(axis=0)
            velocities = np.cross((-2.0, 1.0, 3.0), relative) + (0.02, -0.03, 0.04) + 2.0 * relative
            entity.set_particles_vel(velocities)
            if is_fixed:
                entity.fix_particles([0, 1, 2])
                entity.set_particles_vel((0.02, -0.03, 0.04), particles_idx_local=[0, 1, 2])
        scene.step()
        positions = tensor_to_array(damping_entities[0].get_particles_pos())
        velocities = tensor_to_array(damping_entities[0].get_particles_vel())
        relative = positions - positions.mean(axis=-2, keepdims=True)
        linear = velocities.mean(axis=-2, keepdims=True)
        angular = np.cross(relative, velocities).sum(axis=-2)
        covariance = np.swapaxes(relative, -2, -1) @ relative
        inertia = np.trace(covariance, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) - covariance
        omega = np.linalg.solve(inertia, angular[..., None])[..., 0]
        rigid_velocities = linear + np.cross(omega[..., None, :], relative)
        for entity in damping_entities[1:]:
            decay = math.exp(-entity.material.damping * scene.dt)
            expected = rigid_velocities + decay * (velocities - rigid_velocities)
            if is_fixed:
                expected = velocities.copy()
                expected[..., 3, :] *= decay
            offset = entity.init_particles - damping_entities[0].init_particles
            assert_allclose(entity.get_particles_pos(), positions + offset, atol=2e-7)
            damped = tensor_to_array(entity.get_particles_vel())
            assert_allclose(damped, expected, atol=2e-6)
            if not is_fixed:
                assert_allclose(damped.mean(axis=-2, keepdims=True), linear, atol=2e-6)
                assert_allclose(np.cross(relative, damped).sum(axis=-2), angular, atol=1e-7)
                energy = ((velocities - rigid_velocities) ** 2).sum(axis=(-2, -1))
                damped_energy = ((damped - rigid_velocities) ** 2).sum(axis=(-2, -1))
                assert (energy > 1e-4).all()
                assert_allclose(damped_energy, decay**2 * energy, atol=1e-7)


# Note that "session" scope must NOT be used because the material while be altered without copy when building the scene
@pytest.fixture(scope="function")
def pbd_material():
    """Fixture for common FEM material properties"""
    return gs.materials.PBD.Elastic()


@pytest.mark.required
def test_maxvolume(pbd_material, show_viewer, box_obj_path):
    scene = gs.Scene(
        pbd_options=gs.options.PBDOptions(
            particle_size=0.1,
        ),
        show_viewer=show_viewer,
    )

    # Mesh without any maximum-element-volume constraint
    pbd1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            verbose=1,
        ),
        material=pbd_material,
    )

    # Mesh with maximum element volume limited to 0.001
    pbd2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            maxvolume=0.001,
            verbose=1,
        ),
        material=pbd_material,
    )

    assert pbd1.n_elems < pbd2.n_elems, (
        f"Mesh with maxvolume=0.01 generated {pbd2.n_elems} elements; "
        f"expected more than {pbd1.n_elems} elements without a volume limit."
    )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_get_mass(n_envs, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 0, 0.5),
            scale=1.0,
        ),
        material=gs.materials.PBD.Cloth(),
    )
    scene.build(n_envs=n_envs)

    mass = cloth.get_mass()
    expected_n = max(n_envs, 1)
    assert mass.shape == (expected_n,), f"Expected shape ({expected_n},), got {mass.shape}"
    assert (mass > 0).all(), f"Expected positive mass, got {mass}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("material_type", [gs.materials.PBD.Cloth])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cloth_attach_fixed_point(n_envs, material_type, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cloth_1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 2.0, 0.5),
            scale=1.0,
        ),
        material=material_type(),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 0.2, 1.0),
            vis_mode="visual",
        ),
    )
    cloth_2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 0, 0.1),
            scale=2.0,
        ),
        material=material_type(),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 0.2, 1.0),
            vis_mode="visual",
        ),
    )
    scene.build(n_envs=n_envs)

    # Make sure that 'get_position', 'get_velocity' is working
    for cloth in (cloth_1, cloth_2):
        init_com = torch.tensor([2.0, 0.0, 1.0])
        cloth.set_position(init_com)
        cloth.process_input()
        poss = cloth.get_particles_pos().clone()
        assert_allclose(poss.mean(dim=-2), init_com, tol=1e-2)
        poss += torch.tensor([0.5, -1.0, -0.5])
        vels = torch.rand_like(poss)
        cloth.set_position(poss)
        cloth.set_velocity(vels)
        assert_allclose(cloth.get_particles_vel(), 0.0, tol=tol)
        cloth.process_input()
        assert_allclose(cloth.get_particles_pos(), poss, tol=tol)
        assert_allclose(cloth.get_particles_vel(), vels, tol=tol)
    scene.reset()

    # Simulate for a while
    for _ in range(40):
        scene.step()

    # Make sure that the cloth is landing perfectly vertically and laying on the ground without moving
    poss = cloth_2.get_particles_pos()
    assert_allclose(poss[..., :2], cloth_2._mesh.verts[..., :2], tol=tol)
    assert_allclose(poss[..., 2], cloth_2._particle_size / 2, tol=tol)
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels, 0.0, tol=tol)

    # Attach top-left corner and simulate for a while
    particle_idx = cloth_2.find_closest_particle((-1, -1, 0))
    particle_pos_ref = (-0.5, -0.5, 0.05)
    cloth_2.set_particles_pos(particle_pos_ref, particle_idx)
    cloth_2.fix_particles(particle_idx)
    for i in range(60):
        scene.step()

    # Make sure that the corner is at the target position, some points are still on the ground, and none are moving
    poss = cloth_2.get_particles_pos()
    if scene.n_envs > 0:
        particle_pos = poss[torch.arange(scene.n_envs), particle_idx]
    else:
        particle_pos = poss[particle_idx]
    assert_allclose(particle_pos, particle_pos_ref, tol=tol)
    assert_allclose(poss[..., 2].min(dim=-1).values, cloth_2._particle_size / 2, tol=tol)
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels[..., 2].mean(dim=-1), 0.0, tol=1e-3)

    # Release cloth
    cloth_2.release_particle(particle_idx)
    for i in range(30):
        scene.step()

    # Make sure that the cloth is laying on the ground without moving
    poss = cloth_2.get_particles_pos()
    assert_allclose(poss[..., 2], cloth_2._particle_size / 2, tol=tol)
    assert -0.6 < poss[..., :1].min() and poss[..., :1].max() < 0.6
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels[..., 2].mean(dim=-1), 0.0, tol=tol)


@pytest.mark.required
def test_cloth_attach_rigid_link(show_viewer):
    particle_size = 0.01
    box_height = 2.25

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-2,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=particle_size,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 2.0, 4.0),
            camera_lookat=(-1.0, -1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.25, 0.25, box_height),
            size=(0.25, 0.25, 0.25),
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0.25, 0.25, box_height + 0.125 + particle_size),
            scale=0.5,
        ),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )
    scene.build(n_envs=2)

    particles_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    box_link_idx = box.links[0].idx
    cloth.fix_particles_to_link(box_link_idx, particles_idx_local=particles_idx)

    # leftward velocity for both envs on base linear DOFs [3,4,5]
    vel = np.array([[-0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    box.set_dofs_velocity(vel, dofs_idx_local=[0, 1, 2])

    cloth_pos0 = cloth.get_particles_pos()[:, particles_idx].clone()
    link_pos0 = scene.rigid_solver.links[box_link_idx].get_pos().clone()

    for _ in range(20):
        scene.step()

    # wait for the box to stop
    box.set_dofs_velocity(np.zeros_like(vel), dofs_idx_local=[0, 1, 2])
    for _ in range(10):
        scene.step()

    # Check that the attached particles followed the link displacement per env
    cloth_pos1 = cloth.get_particles_pos()[:, particles_idx].clone()
    link_pos1 = scene.rigid_solver.links[box_link_idx].get_pos().clone()
    assert_allclose((cloth_pos1 - cloth_pos0).movedim(0, -2), link_pos1 - link_pos0, atol=2e-5)

    # Release cloth and restore box's speed
    box.set_dofs_velocity(vel, dofs_idx_local=[0, 1, 2])
    cloth.release_particle(particles_idx)
    for _ in range(15):
        scene.step()

    # Make sure that the cloth is laying on the ground without moving
    cloth_pos2 = cloth.get_particles_pos()[:, particles_idx]
    link_pos2 = scene.rigid_solver.links[box_link_idx].get_pos()
    link_disp = link_pos2 - link_pos1
    cloth_disp = cloth_pos2 - cloth_pos1
    assert ((cloth_disp.movedim(0, -2) - link_disp).norm(dim=-1) > 0.2).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("is_regular_grid", [False, True])
@pytest.mark.parametrize("options_type", [gs.options.PBDOptions, gs.options.PBDUnifiedOptions])
def test_one_way_rigid_surface_collision(n_envs, is_regular_grid, options_type, show_viewer):
    pbd_options = options_type(
        particle_size=0.08,
        lower_bound=(-0.5, -0.5, 0.0),
        upper_bound=(0.5, 0.5, 1.0),
    )
    if isinstance(pbd_options, gs.options.PBDUnifiedOptions):
        pbd_options.max_solver_iterations = 1
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbd_options=pbd_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -1.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    if is_regular_grid:
        vertices, elements = create_tetrahedral_grid(
            lower=(-0.2, -0.2, -0.2), upper=(0.2, 0.2, 0.2), resolution=(2, 2, 2)
        )
        morph = gs.morphs.TetrahedralMesh(
            pos=(0.1, -0.05, 0.5),
            euler=(0.0, 0.0, 90.0),
            offset_pos=(0.05, 0.1, 0.0),
            vertices=vertices,
            elements=elements,
        )
    else:
        morph = gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.4, 0.4, 0.4),
        )
    elastic = scene.add_entity(
        morph=morph,
        material=gs.materials.PBD.Elastic(),
        vis_mode=None if is_regular_grid else "collision",
    )
    particles_initial = elastic.init_particles
    if is_regular_grid:
        assert_allclose(particles_initial, vertices[:, (1, 0, 2)] * (-1, 1, 1) + (0.0, 0.0, 0.5), atol=1e-7)
    triangles_idx = elastic.surface_triangles
    triangles_initial = particles_initial[triangles_idx]
    vverts_rest_offset = elastic.vmesh.verts[:, None] - particles_initial[None]
    vverts_particles_idx = np.linalg.norm(vverts_rest_offset, axis=-1).argmin(axis=1)
    contact_triangles = []
    contact_weights = []
    contact_positions = []
    rigid_entities = []
    vertex_rigid_entities = []
    for direction, is_edge in ((1, False), (-1, True)):
        side_triangles_idx = np.flatnonzero((direction * triangles_initial[..., 0] > 0.2 - 1e-6).all(axis=1))
        if is_edge:
            weights = np.array(
                (
                    (0.5, 0.5, 0.0),
                    (0.0, 0.5, 0.5),
                    (0.5, 0.0, 0.5),
                )
            )
        else:
            weights = np.array(((1 / 3,) * 3,))
        candidates = np.einsum("kv,fvd->fkd", weights, triangles_initial[side_triangles_idx]).reshape((-1, 3))
        distances = np.linalg.norm(candidates[:, None] - particles_initial[None], axis=-1).min(axis=1)
        candidate_idx = np.argmax(distances)
        half_size = (distances[candidate_idx] - scene.pbd_options.particle_size / 2) / 4
        assert half_size > 0.0
        triangle_idx = side_triangles_idx[candidate_idx // len(weights)]
        if not is_edge:
            triangle = triangles_initial[triangle_idx]
            edge_normals = np.cross(triangle[(1, 2, 0), :] - triangle, (direction, 0, 0))
            edge_distances = np.abs(np.einsum("ij,ij->i", edge_normals, candidates[candidate_idx] - triangle))
            assert (edge_distances > half_size * np.abs(edge_normals).sum(axis=1)).all()
        contact_triangles.append(triangles_idx[triangle_idx])
        contact_weights.append(weights[candidate_idx % len(weights)])
        contact_positions.append(candidates[candidate_idx])
        initial_pos = candidates[candidate_idx].copy()
        initial_pos[0] += direction * (half_size + scene.pbd_options.particle_size / 2 + 0.01)
        rigid_entities.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=initial_pos,
                    size=(2 * half_size,) * 3,
                ),
                material=gs.materials.Rigid(
                    is_coup_reaction_enabled=False,
                ),
            )
        )
        # Wide, offset faces keep rigid vertices and face diagonals clear of the elastic surface.
        vertex_rigid_entities.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(direction, 0.8, 0.5),
                    size=(2 * half_size, 3.0, 3.0),
                ),
                material=gs.materials.Rigid(
                    is_coup_reaction_enabled=False,
                ),
            )
        )
        box_dist = np.abs(particles_initial - candidates[candidate_idx]) - half_size
        assert (np.linalg.norm(np.maximum(box_dist, 0.0), axis=-1) > scene.pbd_options.particle_size / 2).all()

    scene.build(n_envs=n_envs)
    for rigid, contact_pos in zip(rigid_entities, contact_positions):
        if isinstance(pbd_options, gs.options.PBDUnifiedOptions):
            velocity = np.zeros(6)
            velocity[:3] = (contact_pos - rigid.morph.pos) / scene.dt
            rigid.set_dofs_velocity(velocity, envs_idx=n_envs - 1 if n_envs else None)
        else:
            rigid.set_pos(contact_pos, envs_idx=n_envs - 1 if n_envs else None)
    scene.step()
    particles_pos = tensor_to_array(elastic.get_particles_pos()).reshape((-1, elastic.n_particles, 3))
    vverts_pos, _, vfaces = scene.pbd_solver.get_state_render()
    vverts_pos = qd_to_numpy(vverts_pos, transpose=True)
    if isinstance(pbd_options, gs.options.PBDOptions):
        render_mesh = trimesh.Trimesh(vertices=vverts_pos[-1], faces=qd_to_numpy(vfaces, transpose=True), process=False)
        _, contact_distances, _ = trimesh.proximity.closest_point(render_mesh, np.stack(contact_positions))
        assert (contact_distances > np.array([rigid.morph.size[0] / 2 for rigid in rigid_entities])).all()
    assert_equal(vverts_pos, particles_pos[:, vverts_particles_idx])
    for contact_idx, (direction, rigid) in enumerate(zip((1, -1), rigid_entities)):
        contact_vertices = particles_pos[-1, contact_triangles[contact_idx]]
        contact_pos = (contact_vertices * contact_weights[contact_idx][:, None]).sum(axis=0)
        if isinstance(pbd_options, gs.options.PBDOptions):
            assert direction * (contact_pos[0] - contact_positions[contact_idx][0]) < -rigid.morph.size[0] / 2
        else:
            assert direction * (contact_pos[0] - contact_positions[contact_idx][0]) < 0.0
        assert_allclose(
            rigid.get_pos(envs_idx=n_envs - 1 if n_envs else None), contact_positions[contact_idx], atol=1e-6
        )
        if isinstance(pbd_options, gs.options.PBDUnifiedOptions):
            assert_allclose(
                rigid.get_vel(envs_idx=n_envs - 1 if n_envs else None),
                (contact_positions[contact_idx] - rigid.morph.pos) / scene.dt,
                atol=1e-6,
            )
        else:
            assert_equal(rigid.get_vel(), 0.0)
    if n_envs:
        assert_allclose(particles_pos[0], particles_initial, atol=1e-6)

    scene.reset()
    scene.step()
    particles_pos = tensor_to_array(elastic.get_particles_pos()).reshape((-1, elastic.n_particles, 3))
    assert_allclose(particles_pos, particles_initial[None], atol=1e-6)
    vverts_pos, _, _ = scene.pbd_solver.get_state_render()
    assert_equal(qd_to_numpy(vverts_pos, transpose=True), particles_pos[:, vverts_particles_idx])

    scene.reset()
    contact_vertices_idx = []
    contact_positions.clear()
    for direction, rigid in zip((1, -1), vertex_rigid_entities):
        i_v = np.argmax(direction * particles_initial[:, 0])
        contact_vertices_idx.append(i_v)
        contact_pos = particles_initial[i_v].copy()
        contact_pos[0] += direction * 0.75 * rigid.morph.size[0] / 2
        contact_pos[1:] = rigid.morph.pos[1:]
        contact_positions.append(contact_pos)
        rigid.set_pos(contact_pos, envs_idx=n_envs - 1 if n_envs else None)
    scene.step()
    particles_pos = tensor_to_array(elastic.get_particles_pos()).reshape((-1, elastic.n_particles, 3))
    for direction, rigid, i_v, contact_pos in zip(
        (1, -1), vertex_rigid_entities, contact_vertices_idx, contact_positions
    ):
        vertices = tensor_to_array(rigid.get_verts()).reshape((-1, rigid.n_verts, 3))[-1]
        assert direction * particles_pos[-1, i_v, 0] <= (direction * vertices[:, 0]).min()
        is_outside = (particles_pos[-1] <= vertices.min(axis=0)) | (particles_pos[-1] >= vertices.max(axis=0))
        assert is_outside.any(axis=-1).all()
        assert_allclose(rigid.get_pos(envs_idx=n_envs - 1 if n_envs else None), contact_pos, atol=1e-6)
        assert_equal(rigid.get_vel(), 0.0)
    if isinstance(pbd_options, gs.options.PBDUnifiedOptions):
        assert_allclose(particles_pos[..., 1:], particles_initial[None, :, 1:], atol=1e-6)
    if n_envs:
        assert_allclose(particles_pos[0], particles_initial, atol=1e-6)

    if isinstance(pbd_options, gs.options.PBDUnifiedOptions):
        scene.reset()
        positions = particles_initial.copy()
        i_v = np.argmax(positions[:, :2].sum(axis=-1))
        positions[i_v, :2] = np.array(pbd_options.upper_bound[:2]) + 0.05
        elastic.set_particles_pos(positions, envs_idx=n_envs - 1 if n_envs else None)
        scene.pbd_solver.project_collision()
        positions[i_v, :2] = pbd_options.upper_bound[:2]
        particles_pos = tensor_to_array(elastic.get_particles_pos()).reshape((-1, elastic.n_particles, 3))
        assert_allclose(particles_pos[-1], positions, atol=1e-7)
        if n_envs:
            assert_allclose(particles_pos[0], particles_initial, atol=1e-7)


@pytest.mark.required
@pytest.mark.precision("32")
@pytest.mark.parametrize("n_envs", [0, 2])
def test_solid_fracture_and_plasticity(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=5,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.005,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 0.6),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -0.8, 0.5),
            camera_lookat=(0.05, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(),
    )
    # bar snapped onto a corner: two rigid fragments held by one breakable seam
    bar = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-0.01, -0.01, 0.12),
            upper=(0.01, 0.01, 0.24),
            quat=(0.9239, 0.0, 0.3827, 0.0),
        ),
        material=gs.materials.PBD.Solid(
            sampler="regular",
            stiffness=1.0,
            fracture_threshold=0.05,
            fragment_seeds=[(0.0, 0.0, -0.025), (0.0, 0.0, 0.025)],
        ),
    )
    plastic = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(0.60, -0.04, 0.2),
            upper=(0.68, 0.04, 0.28),
        ),
        material=gs.materials.PBD.Solid(
            sampler="regular",
            stiffness=1.0,
            # perfectly-plastic contact (plastic_creep > 0) bounds the landing compression to roughly
            # vel * substep_dt, so the yield threshold must stay below that for the box to dent
            yield_threshold=0.1,
            plastic_creep=0.05,
        ),
    )
    camera = scene.add_camera(
        res=(128, 128),
        pos=(0.3, -0.5, 0.35),
        lookat=(0.05, 0.0, 0.2),
        near=0.01,
    )
    scene.build(n_envs=n_envs)
    solver = scene.pbd_solver
    initial_snapshot = scene.get_state()
    n_bonds = solver.n_bonds
    z0 = tensor_to_array(bar.get_particles_pos()).reshape((-1, bar.n_particles, 3))[..., 2].mean(axis=-1)

    # the fragments' render meshes are closed and their volumes sum to the original box volume
    vverts_rest = bar._vverts.reshape((-1, 3, 3))
    frag_of_tri = bar._vvert_fragments.reshape((-1, 3))[:, 0]
    vol_sum = 0.0
    for f in np.unique(frag_of_tri):
        tris_f = vverts_rest[frag_of_tri == f]
        keys, inverse = np.unique(tris_f.reshape(-1, 3), axis=0, return_inverse=True)
        welded = keys[inverse.reshape(-1, 3)]
        vol_sum += np.einsum("ij,ij->i", welded[:, 0], np.cross(welded[:, 1], welded[:, 2])).sum() / 6.0
        welded_edges = np.sort(inverse.reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        assert (np.unique(welded_edges, axis=0, return_counts=True)[1] == 2).all()
    assert_allclose(vol_sum, 0.02 * 0.02 * 0.12, rtol=0.02, atol=1e-9)

    # the displayed mesh follows particle edits and scene reset without needing a step
    bar_vv = slice(bar.vvert_start, bar.vvert_start + bar.n_vverts)
    v_rest = qd_to_numpy(solver.get_state_render()[0], transpose=True)[:, bar_vv].copy()
    cap_faces = bar._vfaces[bar._vvert_seams[bar._vfaces[:, 0]] >= 0]
    assert cap_faces.shape[0] > 0
    cap_triangles = v_rest[:, cap_faces]
    cap_edges = cap_triangles[..., 1:, :] - cap_triangles[..., 0, None, :]
    assert_equal(np.linalg.norm(np.cross(cap_edges[..., 0, :], cap_edges[..., 1, :]), axis=-1), 0.0)
    image_rest = camera.render()[0]
    bar.set_particles_pos(tensor_to_array(bar.get_particles_pos()) + np.array([0.05, 0.0, 0.0]))
    v_moved = qd_to_numpy(solver.get_state_render()[0], transpose=True)[:, bar_vv].copy()
    assert_allclose(v_moved - v_rest, np.array([[0.05, 0.0, 0.0]]), atol=1e-6)
    assert not np.array_equal(camera.render()[0], image_rest)
    scene.reset()
    v_reset = qd_to_numpy(solver.get_state_render()[0], transpose=True)[:, bar_vv].copy()
    assert_allclose(v_reset, v_rest, atol=1e-6)
    assert_equal(camera.render()[0], image_rest)

    # Uneven overlapping memberships must conserve mass-weighted translation in one projection.
    memberships = qd_to_numpy(solver.particle_cluster_ptr.member_count)
    assert np.ptp(memberships[plastic.particle_start : plastic.particle_end]) > 0
    p_plastic = tensor_to_array(plastic.get_particles_pos())
    deformation = np.zeros_like(p_plastic)
    deformation[..., 0] = 0.003 * np.sin(np.arange(plastic.n_particles))
    plastic.set_particles_pos(p_plastic + deformation)
    p_before = tensor_to_array(solver.get_state(0).pos).astype(np.float64)
    mass = qd_to_numpy(solver.particles_info.mass).astype(np.float64)
    solver._kernel_solve_shape_matching(0)
    p_after = tensor_to_array(solver.get_state(0).pos).astype(np.float64)
    com_shift = ((p_after - p_before) * mass[:, None]).sum(axis=-2) / mass.sum()
    assert_allclose(com_shift, 0.0, atol=2e-7)
    scene.reset(initial_snapshot)

    # An unloaded stationary solid must retain its seams and cannot acquire systematic velocity.
    solver.set_gravity((0.0, 0.0, 0.0))
    p_static = tensor_to_array(solver.get_state(0).pos).copy()
    for _ in range(20):
        scene.step()
    assert_allclose(tensor_to_array(solver.get_state(0).pos), p_static, atol=5e-6)
    assert_allclose(tensor_to_array(solver.get_state(0).vel), 0.0, atol=1e-3)
    assert qd_to_numpy(solver.bonds_state.alive, transpose=True).all()
    assert (qd_to_numpy(solver.seams_state.damage, transpose=True) == 0.0).all()
    scene.reset(initial_snapshot)
    solver.set_gravity((0.0, 0.0, -9.81))

    for _ in range(30):
        scene.step()

    # free fall: the seam holds and the centroid follows the analytic trajectory
    assert qd_to_numpy(solver.bonds_state.alive, transpose=True).all()
    assert (qd_to_numpy(solver.seams_state.damage, transpose=True) == 0.0).all()
    assert not qd_to_numpy(solver.seams_state.is_dead, transpose=True).any()
    z_fall = tensor_to_array(bar.get_particles_pos()).reshape((-1, bar.n_particles, 3))[..., 2].mean(axis=-1)
    assert_allclose(z0 - z_fall, 0.5 * 9.81 * (30 * scene.dt) ** 2, atol=2e-3)

    bar.set_particles_vel((0.0, 0.0, -4.0))
    # the box re-impacts inside the same window: the perfectly-plastic contact absorbs its free-fall
    # landing, and a fresh slam keeps the rest-shape creep measurable
    plastic.set_particles_vel((0.0, 0.0, -4.0))
    for _ in range(80):
        scene.step()

    # the slam snaps the bar: the seam opens fully and the two fragments separate
    alive = qd_to_numpy(solver.bonds_state.alive, transpose=True)
    assert (alive.sum(axis=-1) < n_bonds).all()
    alive_count = qd_to_numpy(solver.seams_state.alive_count, transpose=True)
    seams_dead = qd_to_numpy(solver.seams_state.is_dead, transpose=True)
    assert (alive_count == 0).any(axis=-1).all()
    # Fully broken seams record death atomically; displayed caps also require separated components.
    assert (seams_dead == (alive_count == 0)).all()
    frag = qd_to_numpy(solver.particles_info.fragment)
    pos = tensor_to_array(bar.get_particles_pos()).reshape((-1, bar.n_particles, 3))
    bar_frag = frag[bar.particle_start : bar.particle_end]
    frag_ids = np.unique(bar_frag)
    frag_cm = [pos[:, bar_frag == f].mean(axis=-2) for f in frag_ids]
    assert (np.linalg.norm(frag_cm[0] - frag_cm[1], axis=-1) > 0.1).all()

    # fragments stay rigid: no rendered face stretched, no tunneling, no NaN
    assert np.isfinite(pos).all()
    assert (pos[..., 2].min(axis=-1) >= -1e-3).all()
    vverts_pos, _, _ = solver.get_state_render()
    vverts_pos = qd_to_numpy(vverts_pos, transpose=True)
    bar_vv = vverts_pos[:, bar.vvert_start : bar.vvert_start + bar.n_vverts]
    cap_triangles = bar_vv[:, cap_faces]
    cap_edges = cap_triangles[..., 1:, :] - cap_triangles[..., 0, None, :]
    assert (np.linalg.norm(np.cross(cap_edges[..., 0, :], cap_edges[..., 1, :]), axis=-1) > 0.0).all()
    e_rest = np.linalg.norm(bar._vverts[bar._vfaces[:, 1]] - bar._vverts[bar._vfaces[:, 0]], axis=-1)
    e_now = np.linalg.norm(bar_vv[..., bar._vfaces[:, 1], :] - bar_vv[..., bar._vfaces[:, 0], :], axis=-1)
    assert (e_now / np.maximum(e_rest, 1e-12) < 1.5).all()

    # the plastic box dented without breaking: rest shape crept measurably
    rest = qd_to_numpy(solver.particles_solid_rest, transpose=True)
    rest_drift = np.abs(rest[:, plastic.particle_start : plastic.particle_end] - plastic.init_particles[None]).max(
        axis=(-2, -1)
    )
    assert (rest_drift > 1e-4).all()
    pos_plastic = tensor_to_array(plastic.get_particles_pos()).reshape((-1, plastic.n_particles, 3))
    assert np.isfinite(pos_plastic).all()
    assert (pos_plastic[..., 2].min(axis=-1) >= -1e-3).all()
    extent_plastic = pos_plastic[..., 2].max(axis=-1) - pos_plastic[..., 2].min(axis=-1)
    assert (extent_plastic < 0.09).all()

    deformed_snapshot = scene.get_state()
    deformed_state = solver.get_state(0)
    scene.reset(initial_snapshot)
    assert_equal(
        solver.get_state(0).solid_rest[:, plastic.particle_start : plastic.particle_end], plastic.init_particles
    )
    deformed_snapshot.serializable()
    scene.reset(deformed_snapshot)
    assert_equal(solver.get_state(0).solid_rest, deformed_state.solid_rest)
    assert_equal(solver.get_state(0).cluster_rest_cm, deformed_state.cluster_rest_cm)
    assert_equal(solver.get_state(0).seams_is_dead, deformed_state.seams_is_dead)

    # reset restores the intact bodies, and they survive a few more steps unbroken
    scene.reset(initial_snapshot)
    assert qd_to_numpy(solver.bonds_state.alive, transpose=True).all()
    assert not qd_to_numpy(solver.seams_state.is_dead, transpose=True).any()
    restored_vertices = qd_to_numpy(solver.get_state_render()[0], transpose=True)[:, bar.vvert_start : bar.vvert_end]
    cap_triangles = restored_vertices[:, cap_faces]
    cap_edges = cap_triangles[..., 1:, :] - cap_triangles[..., 0, None, :]
    assert_equal(np.linalg.norm(np.cross(cap_edges[..., 0, :], cap_edges[..., 1, :]), axis=-1), 0.0)
    pos_reset = tensor_to_array(bar.get_particles_pos()).reshape((-1, bar.n_particles, 3))
    assert_allclose(pos_reset[..., 2].mean(axis=-1), z0, atol=1e-6)
    for _ in range(10):
        scene.step()
    assert qd_to_numpy(solver.bonds_state.alive, transpose=True).all()

    snapshot = scene.get_state()
    rest_snapshot = solver.get_state(0)
    pos_snapshot = bar.get_particles_pos().clone()
    render_snapshot = qd_to_numpy(solver.get_state_render()[0], transpose=True).copy()
    bar_vverts = slice(bar.vvert_start, bar.vvert_start + bar.n_vverts)
    envs_idx = n_envs - 1 if n_envs else None
    offset = torch.tensor([0.05, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
    bar.set_particles_pos(bar.get_particles_pos(envs_idx=envs_idx) + offset, envs_idx=envs_idx)
    render_moved = qd_to_numpy(solver.get_state_render()[0], transpose=True).copy()
    assert_allclose(render_moved[-1, bar_vverts] - render_snapshot[-1, bar_vverts], [0.05, 0.0, 0.0], atol=1e-6)
    if n_envs:
        assert_equal(render_moved[0], render_snapshot[0])

    snapshot.serializable()
    scene.reset(snapshot, envs_idx=envs_idx)
    state_restored = solver.get_state(0)
    assert_allclose(bar.get_particles_pos(), pos_snapshot, atol=1e-6)

    assert_allclose(qd_to_numpy(solver.get_state_render()[0], transpose=True), render_snapshot, atol=1e-6)
    assert_equal(state_restored.solid_rest, rest_snapshot.solid_rest)
    assert_equal(state_restored.cluster_rest_cm, rest_snapshot.cluster_rest_cm)
    assert_equal(state_restored.bonds_alive, rest_snapshot.bonds_alive)
    assert_equal(state_restored.seams_alive_count, rest_snapshot.seams_alive_count)
    assert_equal(state_restored.seams_damage, rest_snapshot.seams_damage)
    assert_equal(state_restored.seams_is_dead, rest_snapshot.seams_is_dead)

    bar.set_particles_pos([float("nan"), 0.0, 0.2], envs_idx=envs_idx)
    with pytest.raises(gs.GenesisException, match="non-finite"):
        solver.check_errno()
    with pytest.raises(gs.GenesisException, match="non-finite"):
        scene.step()
    scene.reset(snapshot, envs_idx=envs_idx)
    solver.check_errno()
    assert_allclose(bar.get_particles_pos(), pos_snapshot, atol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("contact", ["table", "wall"])
def test_solid_contact_friction(n_envs, contact, show_viewer):
    # A single layer of particles removes tipping from the sliding and sticking checks.
    is_table = contact == "table"
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 256.0 if is_table else 0.002,
            substeps=2 if is_table else 1,
            gravity=(0.0, 0.0, -9.81) if is_table else (0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, 0.0)),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.01,
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
        ),
        show_viewer=show_viewer,
    )
    coefficients = ((0.0, 0.0), (0.2, 0.0), (0.2, 1.0), (0.2, 0.8)) if is_table else ((0.0, 0.0), (0.8, 0.8))
    entities = []
    platform = None
    for i, (particle_mu, geom_mu) in enumerate(coefficients):
        y = -0.6 + i * 0.4
        is_moving = is_table and i == 3
        rigid = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, y, 0.15) if is_table else (-0.05, y, 0.4),
                size=(0.9, 0.25, 0.1) if is_table else (0.1, 0.25, 0.8),
                fixed=not is_moving,
            ),
            material=gs.materials.Rigid(rho=1e6, coup_friction=geom_mu, coup_softness=0.0),
        )
        if is_moving:
            platform = rigid
        entities.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0.0, y, 0.225) if is_table else (0.003, y, 0.3),
                    size=(0.08, 0.08, 0.01) if is_table else (0.01, 0.08, 0.08),
                ),
                material=gs.materials.PBD.Solid(
                    sampler="regular",
                    static_friction=particle_mu,
                    kinetic_friction=particle_mu,
                    fracture_threshold=1.0,
                    fragment_seeds=[(0.0, 0.0, 0.0)],
                ),
            )
        )
    scene.build(n_envs=n_envs)
    if is_table:
        # Preserve the settling duration while using exact binary displacement increments.
        for _ in range(math.ceil(0.075 / scene.dt)):
            scene.step(update_visualizer=False)
        for i, entity in enumerate(entities):
            entity.set_particles_vel((0.0, 0.0, 0.0) if i == 3 else (1.0, 0.0, 0.0))
        platform.set_dofs_velocity([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        for entity in entities:
            entity.set_particles_vel((-0.5, 0.0, 0.5))
    stopping_step = math.ceil(1.0 / (9.81 * scene.dt)) if is_table else None
    for i in range(65 if is_table else 20):
        scene.step(update_visualizer=False)
        if is_table and i + 1 == stopping_step:
            # The surface coefficient sets the stopping time ceil(v / (mu * g * dt)).
            assert_allclose(entities[2].get_particles_vel(), 0.0, atol=1e-5)
    velocities = [
        tensor_to_array(entity.get_particles_vel()).reshape((-1, entity.n_particles, 3)) for entity in entities
    ]
    if is_table:
        assert_allclose(velocities[0], [1.0, 0.0, 0.0], atol=1e-5)
        assert_allclose(velocities[1][..., 0].mean(axis=-1), 1.0 - 0.2 * 9.81 * 65 * scene.dt, atol=2e-4)
        assert_allclose(velocities[2], 0.0, atol=1e-5)
        platform_velocity = tensor_to_array(platform.get_vel()).reshape((-1, 3))
        assert_allclose(velocities[3][..., 0].mean(axis=-1), platform_velocity[:, 0], atol=1e-5)
    else:
        # Each particle preserves tangential motion at zero friction, including zero torque.
        assert_allclose(velocities[0][..., 2], 0.5, atol=1e-5)
        assert_allclose(velocities[1][..., 2], 0.0, atol=1e-5)
