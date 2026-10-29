import numpy as np
import pytest
import torch

import trimesh

import genesis as gs

from ..utils import assert_allclose, assert_equal


@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("has_shell", [False, True])
def test_vertex_controls_and_state(n_envs, has_shell, show_viewer):
    dt = 0.005
    gravity = (0.4, -0.7, -9.81)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
            gravity=gravity,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -1.0, 0.7),
            camera_lookat=(0.2, 0.05, 0.15),
        ),
        show_viewer=show_viewer,
    )
    free = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            vertices=((0.0, 0.0, 0.1), (0.1, 0.0, 0.1), (0.0, 0.1, 0.1), (0.0, 0.0, 0.2)),
            elements=((0, 1, 2, 3),),
        ),
        material=gs.materials.VBD.Elastic(),
    )
    if has_shell:
        translating = scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(
                    trimesh.Trimesh(
                        vertices=(
                            (0.0, 0.0, 0.05),
                            (0.1, 0.0, 0.05),
                            (0.0, 0.1, 0.05),
                            (0.0, 0.0, 0.0512),
                            (0.1, 0.0, 0.0512),
                            (0.0, 0.1, 0.0512),
                            (-0.2, 0.0, 0.05),
                            (-0.1, 0.0, 0.05),
                            (-0.2, 0.1, 0.05),
                        ),
                        faces=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
                        process=False,
                    ),
                ),
            ),
            material=gs.materials.VBD.Cloth(
                E=10000000.0,
            ),
        )
        constrained = scene.add_entity(
            morph=gs.morphs.MeshSet(
                files=(
                    trimesh.Trimesh(
                        vertices=((0.3, 0.0, 0.1), (0.4, 0.0, 0.1), (0.3, 0.1, 0.1)),
                        faces=((0, 1, 2),),
                        process=False,
                    ),
                ),
            ),
            material=gs.materials.VBD.Cloth(),
        )
    else:
        constrained = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                vertices=((0.3, 0.0, 0.1), (0.4, 0.0, 0.1), (0.3, 0.1, 0.1), (0.3, 0.0, 0.2)),
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(),
        )
    constrained.set_vertex_constraints([0])
    scene.build(n_envs=n_envs)
    initial = scene.get_state()
    initial_pos = free.get_state().pos
    if has_shell:
        initial_shell_pos = translating.get_state().pos
    fixed_pos = constrained.get_state().pos[:, 0]
    scene.step()
    expected = np.array(gravity) * (dt / 2) ** 2 * 3
    assert_allclose(free.get_state().pos - initial_pos, np.broadcast_to(expected, initial_pos.shape), atol=2e-6)
    assert_allclose(free.get_state().vel, np.broadcast_to(np.array(gravity) * dt, initial_pos.shape), atol=2e-5)
    if has_shell:
        assert_allclose(
            translating.get_state().pos - initial_shell_pos,
            np.broadcast_to(expected, initial_shell_pos.shape),
            atol=2e-6,
        )
        assert_allclose(
            translating.get_state().vel, np.broadcast_to(np.array(gravity) * dt, initial_shell_pos.shape), atol=2e-5
        )
        shell_fixed_pos = translating.get_state().pos[:, 6:]
        translating.set_vertex_constraints([6, 7, 8])
    assert_equal(constrained.get_state().pos[:, 0], fixed_pos)
    envs_idx = [0] if n_envs else None
    state = free.get_state()
    free.set_vertex_forces([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [1, 1], envs_idx=envs_idx)
    loaded = free.get_state()
    assert_equal(loaded.pos, state.pos)
    assert_equal(loaded.vel, state.vel)
    assert_equal(loaded.forces[0, 1], [3.0, 0.0, 0.0])
    assert_equal(loaded.forces[0, [0, 2, 3]], 0.0)
    if n_envs:
        assert_equal(loaded.forces[1], 0.0)
    free.set_vertex_forces([0.0, 0.0, 0.1], [2], envs_idx=envs_idx)
    assert_equal(free.get_state().forces[0, 1], 0.0)
    free.clear_vertex_forces(envs_idx=envs_idx)
    assert_equal(free.get_state().forces, 0.0)
    velocity = free.get_state().vel
    free.set_vertex_forces([0.01, 0.0, 0.0], [0, 1, 2, 3], envs_idx=envs_idx)
    scene.step()
    if has_shell:
        shell_state = translating.get_state()
        expected = np.array(gravity) * (dt / 2) ** 2 * 10
        assert_allclose(
            shell_state.pos[:, :6] - initial_shell_pos[:, :6],
            np.broadcast_to(expected, initial_shell_pos[:, :6].shape),
            atol=2e-6,
        )
        assert_allclose(
            shell_state.vel[:, :6], np.broadcast_to(np.array(gravity) * dt * 2, shell_state.vel[:, :6].shape), atol=2e-5
        )
        assert_equal(shell_state.pos[:, 6:], shell_fixed_pos)
    acceleration = np.array(gravity)
    acceleration[0] += 0.01 / (free.material.rho * 0.1**3 / 24)
    assert_allclose(free.get_state().vel[0] - velocity[0], np.broadcast_to(acceleration * dt, (4, 3)), atol=2e-5)
    constrained.set_vertex_forces([1.0, 0.0, 0.0], [0])
    scene.step()
    assert_equal(constrained.get_state().pos[:, 0], fixed_pos)
    assert_allclose(free.get_state().vel[0] - velocity[0], np.broadcast_to(acceleration * dt * 2, (4, 3)), atol=4e-5)
    if n_envs:
        preserved = free.get_state().pos[1]
        scene.reset(initial, envs_idx=[0])
        assert_equal(free.get_state().pos[0], initial_pos[0])
        assert_equal(free.get_state().pos[1], preserved)
    scene.reset(initial)
    assert_equal(free.get_state().pos, initial_pos)
    assert_equal(free.get_state().vel, 0.0)
    assert_equal(constrained.get_state().forces, 0.0)
    constrained.remove_vertex_constraints()
    scene.step()
    assert (constrained.get_state().pos[:, 0, 2] < fixed_pos[:, 2]).all()
    scene.reset(initial)
    free.set_vertex_constraints([0, 1, 2, 3], free.init_positions[[0, 2, 1, 3]])
    scene.step()
    assert_allclose(
        free.get_state().pos, np.broadcast_to(free.init_positions[[0, 2, 1, 3]], initial_pos.shape), atol=1e-7
    )
    scene.reset(initial)
    free.set_vertex_forces([np.finfo(gs.np_float).max, 0.0, 0.0], [0])
    with pytest.raises(gs.GenesisException, match="invalid deformation"):
        scene.step()
    with pytest.raises(gs.GenesisException, match="Reset the scene"):
        free.get_state()
    scene.reset()
    assert_equal(free.get_state().pos, initial_pos)
    state = scene.vbd_solver.get_state(f=0)
    state.pos[0, 0, 0] = np.nan
    with pytest.raises(gs.GenesisException, match="finite"):
        scene.vbd_solver.set_state(0, state)
    assert_equal(free.get_state().pos, initial_pos)


@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("is_line_search_enabled", [False, True])
def test_ground_and_self_contact(n_envs, is_line_search_enabled, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            substeps=4,
            gravity=(0, 0, -9.81),
        ),
        vbd_options=gs.options.VBDOptions(
            is_contact_diagnostics_enabled=True,
            is_line_search_enabled=is_line_search_enabled,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.8, -2.8, 2.0),
            camera_lookat=(1.1, 0.3, 0.6),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    entities = []
    for height in (0.0005, 0.0015):
        mesh = trimesh.Trimesh(
            vertices=((0.0, 0.0, height), (0.08, 0.0, height), (0.0, 0.08, height), (0.08, 0.08, height)),
            faces=((0, 1, 2), (1, 3, 2)),
            process=False,
        )
        entities.append(
            scene.add_entity(
                morph=gs.morphs.MeshSet(
                    files=(mesh,),
                ),
                material=gs.materials.VBD.Cloth(),
            )
        )
    sliders = []
    for index, friction in enumerate((0.0, 0.8)):
        mesh = trimesh.Trimesh(
            vertices=((0.0, 0.0, 0.0005), (0.04, 0.0, 0.0005), (0.0, 0.04, 0.0005)),
            faces=((0, 1, 2),),
            process=False,
        )
        sliders.append(
            scene.add_entity(
                morph=gs.morphs.MeshSet(
                    pos=(0.15 + 0.1 * index, 0.0, 0.0),
                    files=(mesh,),
                ),
                material=gs.materials.VBD.Cloth(
                    friction_mu=friction,
                ),
            )
        )
    approaching_entities = []
    for index in range(2):
        if index == 0:
            lower_vertices = ((-0.02, -0.02, 0.0), (0.02, -0.02, 0.0), (0.0, 0.02, 0.0))
            upper_vertices = ((-0.002, -0.002, 0.0), (0.002, -0.002, 0.0), (0.0, 0.002, 0.0))
        else:
            lower_vertices = ((-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), (-0.04, -0.04, 0.0))
            upper_vertices = ((0.0, -0.02, 0.0), (0.0, 0.02, 0.0), (0.04, 0.04, 0.0))
        for height, vertices in ((0.2, lower_vertices), (0.2006, upper_vertices)):
            mesh = trimesh.Trimesh(vertices=vertices, faces=((0, 1, 2),), process=False)
            approaching_entities.append(
                scene.add_entity(
                    morph=gs.morphs.MeshSet(
                        pos=(0.8 + 0.3 * index, -0.3, height),
                        files=(mesh,),
                    ),
                    material=gs.materials.VBD.Cloth(
                        thickness=0.00025,
                    ),
                )
            )
    falling = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(0.5, 0.0, 1.0),
            vertices=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
            elements=((0, 1, 2, 3),),
        ),
        material=gs.materials.VBD.Elastic(
            E=100000000.0,
        ),
    )
    mesh = trimesh.Trimesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (0.0004, 0.0, 0.0),
            (0.0008, 0.0, 0.0),
            (0.0, 0.0004, 0.0),
            (0.0004, 0.0004, 0.0),
            (0.0008, 0.0004, 0.0),
            (0.0, 0.0008, 0.0),
            (0.0004, 0.0008, 0.0),
            (0.0008, 0.0008, 0.0),
        ),
        faces=((0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4), (3, 4, 6), (4, 7, 6), (4, 5, 7), (5, 8, 7)),
        process=False,
    )
    dense = scene.add_entity(
        morph=gs.morphs.MeshSet(
            pos=(0.5, 0.2, 0.1),
            files=(mesh,),
        ),
        material=gs.materials.VBD.Cloth(
            E=1000.0,
            shear_stiffness=0.01,
            bending_model=None,
        ),
    )
    rebounding = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(0.7, 0.0, 0.008),
            vertices=(
                (0.0, 0.0, 0.0),
                (0.04, 0.0, 0.0),
                (0.0, 0.04, 0.0),
                (0.04, 0.04, 0.0),
                (0.0, 0.0, 0.04),
                (0.04, 0.0, 0.04),
                (0.0, 0.04, 0.04),
                (0.04, 0.04, 0.04),
            ),
            elements=((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)),
        ),
        material=gs.materials.VBD.Elastic(
            E=10000.0,
        ),
    )
    obstacle = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            vertices=((1.0, 0.0, 0.15), (1.2, 0.0, 0.15), (1.0, 0.2, 0.15), (1.0, 0.0, 0.149)),
            elements=((0, 2, 1, 3),),
        ),
        material=gs.materials.VBD.Elastic(),
    )
    obstacle.set_vertex_constraints([0, 1, 2, 3])
    crossing = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            vertices=((1.03, 0.03, 0.153), (1.032, 0.03, 0.153), (1.03, 0.032, 0.153), (1.03, 0.03, 0.155)),
            elements=((0, 1, 2, 3),),
        ),
        material=gs.materials.VBD.Elastic(
            E=1000000.0,
        ),
    )
    penetrating = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            vertices=(
                (1.07, 0.03, 0.14995),
                (1.072, 0.03, 0.14995),
                (1.07, 0.032, 0.14995),
                (1.07, 0.03, 0.15195),
            ),
            elements=((0, 1, 2, 3),),
        ),
        material=gs.materials.VBD.Elastic(
            E=1000000.0,
        ),
    )
    near_entities = []
    for index, gap_factor in enumerate((0.5, 2.0)):
        entity = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=(1.11 + index * 0.03, 0.03, 0.15 + gap_factor * scene.vbd_options.contact_d_hat),
                vertices=((0.0, 0.0, 0.0), (0.002, 0.0, 0.0), (0.0, 0.002, 0.0), (0.0, 0.0, 0.002)),
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(
                E=1000000.0,
            ),
        )
        entity.set_vertex_forces([0.0, 0.0, entity.material.rho * 0.002**3 / 24 * 9.81], [0, 1, 2, 3])
        near_entities.append(entity)
    tet_vertices = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    centers = np.indices((7, 6, 3)).reshape(3, 126).T * 0.002 + (-0.006, -0.005, 0.52)
    # Disconnected interior tetrahedra reject nearby surface matches, leaving a distant enclosing surface.
    nested_vertices = np.concatenate(
        (0.6 * tet_vertices - 0.15, (centers[:, None] + 0.0003 * tet_vertices).reshape(504, 3))
    )
    nested = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(1.7, 0.0, 1.0),
            vertices=nested_vertices,
            elements=np.arange(508).reshape(127, 4),
        ),
        material=gs.materials.VBD.Elastic(),
    )
    nested.set_vertex_constraints(np.arange(nested.n_vertices))
    embedded = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(1.7, 0.0, 1.0),
            vertices=0.002 * tet_vertices,
            elements=((0, 1, 2, 3),),
        ),
        material=gs.materials.VBD.Elastic(),
    )
    embedded.set_vertex_constraints([0, 1, 2, 3])
    containment_entities = []
    for index, (scale, volume_ratio) in enumerate(((1.0, 0.005), (0.25, 0.005), (1.0, 0.02), (0.25, 0.02))):
        # Exactly representable edges keep gravity cancellation consistent with the stored reference volume.
        query_length = scale / 8192
        compressed = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=(0.4 + 0.5 * index, 0.7, 0.1),
                vertices=0.4 * scale * tet_vertices,
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(),
        )
        compressed.set_vertex_constraints([0, 1, 2, 3])
        interior = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=compressed.init_positions.mean(axis=0),
                vertices=query_length * tet_vertices,
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(),
        )
        interior.set_vertex_constraints([0, 1, 2, 3])
        vertex_mass = interior.material.rho * query_length**3 / 24
        interior.set_vertex_forces([vertex_mass, 0.0, 9.81 * vertex_mass], [0, 1, 2, 3])
        containment_entities.append((compressed, interior, volume_ratio))
    scene.build(n_envs=n_envs)
    initial_velocity = -2.0
    state = scene.vbd_solver.get_state(f=0)
    state.vel[:, falling.v_start : falling.v_start + falling.n_vertices, 2] = initial_velocity
    state.vel[:, crossing.v_start : crossing.v_start + crossing.n_vertices, 2] = -8.0
    for index, entity in enumerate(approaching_entities):
        state.vel[:, entity.v_start : entity.v_start + entity.n_vertices, 2] = 1.0 if index % 2 == 0 else -1.0
    scene.vbd_solver.set_state(f=0, state=state)
    initial_falling = falling.get_state().pos
    initial_dense = dense.get_state().pos
    initial_near = [entity.get_state().pos for entity in near_entities]
    for _ in range(2):
        scene.step()
    n_substeps = 2 * scene.sim.substeps
    displacement = -9.81 * (scene.dt / scene.sim.substeps) ** 2 * n_substeps * (n_substeps + 1) / 2
    assert_allclose(
        falling.get_state().pos[..., 2] - initial_falling[..., 2],
        displacement + initial_velocity * 2 * scene.dt,
        atol=2e-6,
    )
    assert_allclose(dense.get_state().pos - initial_dense, [0.0, 0.0, displacement], atol=2e-7)
    for index in range(2):
        lower_pos = approaching_entities[2 * index].get_state().pos
        upper_pos = approaching_entities[2 * index + 1].get_state().pos
        lower_edge = lower_pos[:, 1] - lower_pos[:, 0]
        if index == 0:
            normal = torch.linalg.cross(lower_edge, lower_pos[:, 2] - lower_pos[:, 0])
            gap = ((upper_pos - lower_pos[:, :1]) * normal[:, None]).sum(dim=-1)
            gap /= torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        else:
            normal = torch.linalg.cross(lower_edge, upper_pos[:, 1] - upper_pos[:, 0])
            gap = ((upper_pos[:, 0] - lower_pos[:, 0]) * normal).sum(dim=-1)
            gap /= torch.linalg.vector_norm(normal, dim=-1)
        assert (gap >= approaching_entities[2 * index].material.thickness - 2e-7).all()
    assert (penetrating.get_state().pos[..., 2] > 0.14999).all()
    assert (crossing.get_state().pos[..., 2] > 0.1499).all()
    assert (
        near_entities[0].get_state().pos[..., 2].amin(dim=-1)
        > initial_near[0][..., 2].amin(dim=-1) + 0.1 * scene.vbd_options.contact_d_hat
    ).all()
    assert_allclose(near_entities[1].get_state().pos, initial_near[1], atol=2e-7)
    heights, velocities = [], []
    for _ in range(28):
        scene.step()
        state = rebounding.get_state()
        heights.append(state.pos[..., 2].mean(dim=-1))
        velocities.append(state.vel[..., 2].mean(dim=-1))
    heights = torch.stack(heights)
    velocities = torch.stack(velocities)
    assert (velocities[8:].amax(dim=0) > 0.05).all()
    assert (heights[8:].amax(dim=0) - heights.amin(dim=0) > 0.001).all()
    lower = entities[0].get_state()
    upper = entities[1].get_state()
    compression = entities[0].material.thickness / 2 + scene.vbd_options.contact_d_hat - lower.pos[..., 2]
    weight = 2 * 0.08**2 * entities[0].material.thickness * entities[0].material.rho * 9.81
    support = compression.clamp_min(0).sum(dim=-1) * scene.vbd_options.shell_contact_stiffness
    assert (upper.pos[..., 2] < 0.0015).all()
    assert_allclose(lower.vel[..., 2], 0.0, atol=0.002)
    assert_allclose(upper.vel[..., 2], 0.0, atol=0.002)
    state = scene.vbd_solver.get_state(0)
    for entity in sliders:
        state.vel[:, entity.v_start : entity.v_start + entity.n_vertices, 0] = 0.1
    scene.vbd_solver.set_state(0, state)
    for _ in range(20):
        scene.step()
    assert_allclose(sliders[0].get_state().vel[..., 0], 0.1, atol=0.002)
    assert_allclose(sliders[1].get_state().vel[..., 0], 0.0, atol=0.005)
    state = scene.vbd_solver.get_state(0)
    lower_indices = slice(entities[0].v_start, entities[0].v_start + entities[0].n_vertices)
    upper_indices = slice(entities[1].v_start, entities[1].v_start + entities[1].n_vertices)
    state.pos[:, upper_indices] = state.pos[:, lower_indices]
    scene.vbd_solver.set_state(0, state)
    falling_pos = falling.get_state().pos
    for _ in range(2):
        scene.step()
    assert (falling.get_state().pos[..., 2] < falling_pos[..., 2]).all()
    scene.reset()
    assert_allclose(entities[1].get_state().pos[..., 2], 0.0015, atol=1e-7)
    targets = nested.get_state().pos
    targets[:, 4:, 2] -= 0.5
    nested.set_vertex_constraints(np.arange(nested.n_vertices), targets)
    state = scene.vbd_solver.get_state(f=0)
    state.pos[:, nested.v_start : nested.v_start + nested.n_vertices] = targets
    scene.vbd_solver.set_state(f=0, state=state)
    embedded.remove_vertex_constraints()
    initial_embedded = embedded.get_state().pos
    scene.step()
    displacement = -9.81 * scene.dt**2 * (scene.sim.substeps + 1) / (2 * scene.sim.substeps)
    # Exhausting the surface search discards its contact, so gravity determines the embedded entity's motion.
    assert_allclose(embedded.get_state().pos - initial_embedded, [0.0, 0.0, displacement], atol=2e-6)
    state = scene.vbd_solver.get_state(f=0)
    initial_interiors = []
    for compressed, interior, volume_ratio in containment_entities:
        compressed_indices = slice(compressed.v_start, compressed.v_start + compressed.n_vertices)
        interior_indices = slice(interior.v_start, interior.v_start + interior.n_vertices)
        targets = state.pos[:, compressed_indices].clone()
        targets[:, 3, 2] = targets[:, 0, 2] + volume_ratio * (targets[:, 3, 2] - targets[:, 0, 2])
        state.pos[:, compressed_indices] = targets
        state.fixed_pos[:, compressed_indices] = targets
        positions = state.pos[:, interior_indices].clone()
        positions += targets.mean(dim=1, keepdim=True) - positions[:, :1]
        positions[..., 2] -= (targets[:, 3:4, 2] - targets[:, :1, 2]) / 8
        state.pos[:, interior_indices] = positions
        state.is_fixed[:, interior_indices] = False
        initial_interiors.append(positions)
    scene.vbd_solver.set_state(f=0, state=state)
    scene.step()
    displacement = scene.dt**2 * (scene.sim.substeps + 1) / (2 * scene.sim.substeps)
    for (compressed, interior, volume_ratio), initial in zip(containment_entities, initial_interiors):
        if volume_ratio < 0.01:
            assert_allclose(interior.get_state().pos - initial, [displacement, 0.0, 0.0], atol=2e-7)
        else:
            depth = initial[:, 0, 2] - compressed.init_positions[0, 2]
            assert (interior.get_state().pos[..., 2].mean(dim=-1) < initial[..., 2].mean(dim=-1) - depth / 2).all()
    assert_allclose(support, weight, atol=2e-5)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_plastic_unloading_and_bending_switch(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            substeps=2,
            gravity=(0, 0, 0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7, -0.8, 0.9),
            camera_lookat=(0.25, 0.15, 0.05),
        ),
        show_viewer=show_viewer,
    )
    solids, shells = [], []
    for index, (is_plastic, hardening, yield_strain) in enumerate(
        ((False, 0.0, 1.0), (True, 0.0, 1.0), (True, 100.0, 1.0), (True, 0.0, 50.0))
    ):
        solid = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=(index * 0.15, 0.0, 0.0),
                vertices=((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.05)),
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(
                is_volume_plasticity_enabled=is_plastic,
                volume_yield_strain_percent=yield_strain,
                volume_plastic_hardening=hardening,
            ),
        )
        solid.set_vertex_constraints([0, 1, 2, 3])
        solids.append(solid)
    for index, (is_plastic, bending_model, hardening, yield_angle) in enumerate(
        (
            (False, "DiscreteShellBending", 0.0, 10.0),
            (True, "DiscreteShellBending", 0.0, 10.0),
            (True, None, 0.0, 10.0),
            (True, "DiscreteShellBending", 100.0, 10.0),
            (True, "DiscreteShellBending", 0.0, 180.0),
        )
    ):
        mesh = trimesh.Trimesh(
            vertices=((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.05, -0.05, 0.0)),
            faces=((0, 1, 2), (1, 0, 3)),
            process=False,
        )
        shell = scene.add_entity(
            morph=gs.morphs.MeshSet(
                pos=(index * 0.15, 0.25, 0.0),
                files=(mesh,),
            ),
            material=gs.materials.VBD.Cloth(
                bending_model=bending_model,
                bending_rigidity=0.01,
                is_bending_plasticity_enabled=is_plastic,
                bending_yield_angle_deg=yield_angle,
                bending_plastic_hardening=hardening,
            ),
        )
        shell.set_vertex_constraints([0, 1, 2, 3])
        shells.append(shell)
    scene.build(n_envs=n_envs)
    for solid in solids:
        targets = solid.init_positions.copy()
        targets[3, 0] += 0.02
        solid.set_vertex_constraints([0, 1, 2, 3], targets)
    for shell in shells:
        targets = shell.init_positions.copy()
        targets[3, 1] = 0.225
        targets[3, 2] = np.sqrt(3) * 0.025
        shell.set_vertex_constraints([0, 1, 2, 3], targets)
    scene.step()
    for entity in (*solids, *shells):
        entity.remove_vertex_constraints([3])
    for _ in range(40):
        scene.step()
    residual = [entity.get_state().pos[:, 3, 0] - entity.init_positions[3, 0] for entity in solids]
    assert_allclose(residual[0], 0.0, atol=0.001)
    assert (residual[1] > 0.008).all()
    assert (residual[1] > residual[2] + 0.005).all()
    assert_allclose(residual[3], 0.0, atol=0.001)
    heights = [entity.get_state().pos[:, 3, 2] for entity in shells]
    assert_allclose(heights[0], 0.0, atol=0.002)
    assert (heights[1] > 0.02).all()
    assert_allclose(heights[2], np.sqrt(3) * 0.025, atol=0.002)
    assert (heights[1] > heights[3] + 0.02).all()
    assert_allclose(heights[4], 0.0, atol=0.002)
    scene.reset()
    for entity in (*solids, *shells):
        assert_allclose(
            entity.get_state().pos, np.broadcast_to(entity.init_positions, entity.get_state().pos.shape), atol=1e-7
        )
        assert_equal(entity.get_state().vel, 0.0)
    for solid in solids:
        targets = solid.init_positions.copy()
        targets[3, 2] = -0.05
        solid.set_vertex_constraints([3], targets[3])
    scene.step()
    for solid in solids:
        state = solid.get_state()
        assert_allclose(state.pos[:, 3, 2], -0.05, atol=1e-7)
        assert_equal(state.plastic_strain, 0.0)
        assert_allclose(state.plastic_inverse, np.broadcast_to(np.eye(3), state.plastic_inverse.shape), atol=1e-7)
        solid.remove_vertex_constraints([3])
    scene.step()
    for solid in solids:
        assert (solid.get_state().pos[:, 3, 2] > 0.0).all()


@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("is_line_search_enabled", [False, True])
@pytest.mark.parametrize("convergence_tolerance", [0.0, 1e-7])
def test_material_response(n_envs, is_line_search_enabled, convergence_tolerance, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            substeps=2,
            gravity=(0, 0, 0),
        ),
        vbd_options=gs.options.VBDOptions(
            convergence_tolerance=convergence_tolerance,
            shell_ground_stiffness=10.0,
            shell_ground_d_hat=0.0,
            is_line_search_enabled=is_line_search_enabled,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.0, 0.9),
            camera_lookat=(0.4, 0.2, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, -1.0),
        ),
    )
    mesh = trimesh.Trimesh(
        vertices=((0.0, 0.0, -1.001), (0.02, 0.0, -1.001), (0.0, 0.02, -1.001)),
        faces=((0, 1, 2),),
        process=False,
    )
    grounded = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(mesh,),
        ),
        material=gs.materials.VBD.Cloth(
            E=1e-8,
            rho=800.0,
            friction_mu=0.0,
            shear_stiffness=0.0,
            contact_thickness=0.0004,
            bending_model=None,
        ),
    )
    solids, shells, accelerated, recovering = [], [], [], []
    for index, (modulus, poisson) in enumerate(((1e5, 0.25), (1e6, 0.25), (1e5, 0.45))):
        entity = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=(index * 0.2, 0.0, 0.0),
                vertices=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(
                E=modulus,
                nu=poisson,
            ),
        )
        entity.set_vertex_constraints([0, 1, 2])
        entity.set_vertex_forces([0, 0, -0.1], [3])
        solids.append(entity)
    for index, (shear, thickness, damping) in enumerate(
        ((8.0, 0.00025, 0.0), (8000.0, 0.00025, 0.0), (8.0, 0.0005, 0.0), (0.0, 0.00025, 0.0), (8.0, 0.00025, 0.02))
    ):
        mesh = trimesh.Trimesh(
            vertices=((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.0)),
            faces=((0, 1, 2),),
            process=False,
        )
        entity = scene.add_entity(
            morph=gs.morphs.MeshSet(
                pos=(index * 0.2, 0.2, 0.0),
                files=(mesh,),
            ),
            material=gs.materials.VBD.Cloth(
                shear_stiffness=shear,
                thickness=thickness,
                membrane_damping=damping,
            ),
        )
        entity.set_vertex_constraints([0, 1])
        entity.set_vertex_forces([0.001, 0, 0], [2])
        shells.append(entity)
    bending_cases = []
    for index, (bending_model, damping) in enumerate(
        (("DiscreteShellBending", 0.0), ("QuadraticBending", 0.0), ("QuadraticBending", 0.0025))
    ):
        length = 0.001
        mesh = trimesh.Trimesh(
            vertices=((0, 0, 0), (length, 0, 0), (0, length, 0), (length, -length / np.sqrt(2), length / np.sqrt(2))),
            faces=((0, 1, 2), (1, 0, 3)),
            process=False,
        )
        entity = scene.add_entity(
            morph=gs.morphs.MeshSet(
                pos=(index * 0.2, 0.6, 0.0),
                files=(mesh,),
            ),
            material=gs.materials.VBD.Cloth(
                E=1e-8,
                rho=800.0,
                shear_stiffness=0.0,
                contact_thickness=0.0,
                bending_damping=damping,
                bending_model=bending_model,
                bending_rigidity=1e-9,
            ),
        )
        entity.set_vertex_constraints([0, 1, 2])
        bending_cases.append(entity)
    for index, density in enumerate((360.0, 720.0)):
        entity = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=(index * 0.2, 0.4, 0.0),
                vertices=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(
                rho=density,
            ),
        )
        entity.set_vertex_forces([0.01, 0, 0], [0, 1, 2, 3])
        accelerated.append(entity)
    for index, height_ratio in enumerate((-1.0, -0.1, 0.0)):
        entity = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=(0.6 + index * 0.2, 0.0, 0.0),
                vertices=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(),
        )
        entity.set_vertex_constraints([0, 1, 2])
        recovering.append((entity, height_ratio))
    constitutive = []
    for index, model in enumerate(("stable_neohookean", "log_neohookean")):
        entity = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=(0.6 + index * 0.3, 0.25, 0.0),
                vertices=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
                elements=((0, 1, 2, 3),),
            ),
            material=gs.materials.VBD.Elastic(
                model=model,
            ),
        )
        entity.set_vertex_constraints([0, 1, 2])
        constitutive.append(entity)
    scene.build(n_envs=n_envs)
    state = scene.vbd_solver.get_state(f=0)
    for entity, height_ratio in recovering:
        state.pos[:, entity.v_start + 3, 2] = height_ratio * entity.init_positions[3, 2]
    for entity in constitutive:
        state.pos[:, entity.v_start + 3, 2] = 0.5 * entity.init_positions[3, 2]
    scene.vbd_solver.set_state(f=0, state=state)
    scene.step()
    height, velocity = grounded.init_positions[0, 2], 0.0
    substep_dt = scene.dt / scene.sim.substeps
    mass = grounded.material.rho * grounded.material.thickness * 0.02**2 / 6
    inertia = mass / substep_dt**2
    stiffness = scene.vbd_options.shell_ground_stiffness
    for _ in range(scene.sim.substeps):
        target = height + substep_dt * velocity
        updated = (inertia * target - stiffness) / (inertia + stiffness) if target < -1.0 else target
        velocity = (updated - height) / substep_dt
        height = updated
    assert_allclose(grounded.get_state().pos[..., 2], height, atol=2e-7)
    assert_allclose(grounded.get_state().vel[..., 2], velocity, atol=2e-4)
    compression = [entity.init_positions[3, 2] - entity.get_state().pos[:, 3, 2] for entity in solids]
    assert (compression[0] > 0).all()
    assert (compression[1] < compression[0] / 5).all()
    assert (compression[2] < compression[0] * 0.7).all()
    shear_displacement = [entity.get_state().pos[:, 2, 0] - entity.init_positions[2, 0] for entity in shells]
    assert (shear_displacement[0] > 0).all()
    assert (shear_displacement[1] < shear_displacement[0] / 20).all()
    assert (shear_displacement[2] < shear_displacement[0] * 0.8).all()
    assert (shear_displacement[3] > shear_displacement[0]).all()
    assert (shear_displacement[4] < shear_displacement[0] / 2).all()
    for entity in bending_cases:
        material = entity.material
        position = entity.init_positions[3].copy()
        velocity = np.zeros(3)
        if material.bending_model == "QuadraticBending":
            # A single free corner has an isotropic quadratic spring toward the planar cotangent equilibrium.
            length = entity.init_positions[2, 1] - entity.init_positions[0, 1]
            stiffness = 3 * material.bending_rigidity / (2 * length**2)
            mass = material.rho * material.thickness * length**2 / 6
            equilibrium = entity.init_positions[0] + entity.init_positions[1] - entity.init_positions[2]
            substep_dt = scene.dt / scene.sim.substeps
            inertia = mass / substep_dt**2
            damping = stiffness * material.bending_damping / substep_dt
            for _ in range(scene.sim.substeps):
                target = position + substep_dt * velocity
                updated = (inertia * target + stiffness * equilibrium + damping * position) / (
                    inertia + stiffness + damping
                )
                velocity = (updated - position) / substep_dt
                position = updated
        assert_allclose(entity.get_state().pos[:, 3], np.broadcast_to(position, (max(1, n_envs), 3)), atol=2e-7)
        assert_allclose(entity.get_state().vel[:, 3], np.broadcast_to(velocity, (max(1, n_envs), 3)), atol=8e-5)
    for entity in accelerated:
        vertex_mass = entity.material.rho * 0.1**3 / 24
        expected = 0.01 / vertex_mass * scene.dt
        assert_allclose(entity.get_state().vel[..., 0], expected, atol=2e-5)
    for entity, height_ratio in recovering:
        length = entity.init_positions[3, 2]
        material = entity.material
        stiffness = material.E * (1 - material.nu) / ((1 + material.nu) * (1 - 2 * material.nu)) * length / 6
        vertex_mass = material.rho * length**3 / 24
        substep_dt = scene.dt / scene.sim.substeps
        gain = stiffness / (vertex_mass / substep_dt**2 + stiffness)
        initial_height = height_ratio * length
        first_height = (1 - gain) * initial_height + gain * length
        second_height = (1 - gain) * (2 * first_height - initial_height) + gain * length
        assert_allclose(entity.get_state().pos[:, 3, 2], second_height, atol=2e-6)
        assert_allclose(entity.get_state().vel[:, 3, 2], (second_height - first_height) / substep_dt, atol=2e-4)
    material = constitutive[0].material
    length = constitutive[0].init_positions[3, 2]
    mu = material.E / (2 * (1 + material.nu))
    lam = material.E * material.nu / ((1 + material.nu) * (1 - 2 * material.nu))
    vertex_mass = material.rho * length**3 / 24
    substep_dt = scene.dt / scene.sim.substeps
    positions = np.full(len(constitutive), 0.5 * length)
    velocities = np.zeros_like(positions)
    is_log = np.array([entity.material.model == "log_neohookean" for entity in constitutive])
    # With three fixed vertices, uniaxial stress balances inertia in a scalar implicit Euler equation.
    for _ in range(scene.sim.substeps):
        target = positions + substep_dt * velocities
        lower, upper = np.minimum(target, length), np.maximum(target, length)
        for _ in range(40):
            midpoint = (lower + upper) / 2
            stretch = midpoint / length
            stress = np.where(
                is_log,
                mu * (stretch - 1 / stretch) + lam * np.log(stretch) / stretch,
                (lam + 2 * mu) * (stretch - 1),
            )
            residual = vertex_mass / substep_dt**2 * (midpoint - target) + length**2 / 6 * stress
            lower = np.where(residual < 0, midpoint, lower)
            upper = np.where(residual < 0, upper, midpoint)
        updated = (lower + upper) / 2
        velocities = (updated - positions) / substep_dt
        positions = updated
    for index, entity in enumerate(constitutive):
        assert_allclose(entity.get_state().pos[:, 3, 2], positions[index], atol=2e-6)
        assert_allclose(entity.get_state().vel[:, 3, 2], velocities[index], atol=2e-4)
    for height in (0.0, -0.01):
        state = scene.vbd_solver.get_state(f=0)
        for entity in constitutive:
            if entity.material.model == "log_neohookean":
                state.pos[:, entity.v_start + 3, 2] = height
        with pytest.raises(gs.GenesisException, match="invalid deformation"):
            scene.vbd_solver.set_state(f=0, state=state)
        scene.reset()
