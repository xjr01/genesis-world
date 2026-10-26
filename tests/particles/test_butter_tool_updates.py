import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.utils.misc import tensor_to_array


@pytest.mark.parametrize("backend", [gs.gpu])
def test_prescribed_tool_publishes_pose_and_velocity_together(backend):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            disable_constraint=True,
        ),
        show_viewer=False,
    )
    blade = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.02, 0.048, 0.0008),
        ),
        material=gs.materials.Rigid(),
    )
    scene.build()
    rng = np.random.default_rng(17)
    for _ in range(16):
        position = gs.tensor(rng.uniform(-0.1, 0.1, 3))
        quaternion = gs.tensor(Rotation.random(random_state=rng).as_quat()[[3, 0, 1, 2]])
        velocity = gs.tensor(rng.uniform(-1, 1, 6))
        blade.set_pos(position, zero_velocity=False, relative=False)
        blade.set_quat(quaternion, zero_velocity=False, relative=False)
        blade.set_dofs_velocity(velocity)
        expected = [
            tensor_to_array(value) for value in (blade.get_pos(), blade.get_quat(), blade.get_vel(), blade.get_ang())
        ]

        blade.set_pos((0, 0, 0), relative=False)
        blade.set_quat((1, 0, 0, 0), relative=False)
        blade.set_dofs_velocity(velocity, skip_forward=True)
        blade.set_pos(position, zero_velocity=False, relative=False, skip_forward=True)
        blade.set_quat(quaternion, zero_velocity=False, relative=False)
        actual = [
            tensor_to_array(value) for value in (blade.get_pos(), blade.get_quat(), blade.get_vel(), blade.get_ang())
        ]
        for result, reference in zip(actual, expected):
            np.testing.assert_array_equal(result, reference)
        np.testing.assert_allclose(
            tensor_to_array(blade.solver.get_geoms_pos())[0], tensor_to_array(position), atol=1e-7
        )
