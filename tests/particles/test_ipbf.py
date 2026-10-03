import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array
from tests.utils import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_snapshot_restores_movable_pitcher_pose():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=0.1,
            has_boundary_particles=False,
            boundary_pitcher=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Particles(
            positions=((0.0, 0.0, 0.5),),
        ),
        material=gs.materials.IPBF.Liquid(),
    )
    scene.build()
    snapshot = scene.get_state()

    scene.ipbf_solver.set_pitcher_pose(origin=(1.0, 2.0, 3.0), axis=(0.0, 1.0, 0.0))
    changed = scene.ipbf_solver.get_state(0)
    assert_allclose(tensor_to_array(changed.pitcher_origin), (1.0, 2.0, 3.0), atol=1e-12)
    assert_allclose(tensor_to_array(changed.pitcher_axis), (0.0, 1.0, 0.0), atol=1e-12)

    scene.reset(snapshot)
    restored = scene.ipbf_solver.get_state(0)
    assert_allclose(tensor_to_array(restored.pitcher_origin), (0.0, 0.0, 0.0), atol=1e-12)
    assert_allclose(tensor_to_array(restored.pitcher_axis), (0.0, 0.0, 1.0), atol=1e-12)
