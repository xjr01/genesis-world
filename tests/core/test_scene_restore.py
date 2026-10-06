from unittest.mock import Mock

from genesis.engine.couplers import IPCCoupler
from genesis.engine.scene import Scene
from genesis.engine.simulator import Simulator
from genesis.engine.states.solvers import SimState


def test_restore_preserves_registered_initial_state_and_resets_recorders():
    scene = object.__new__(Scene)
    scene._is_built = True
    scene._uid = "test-scene"
    scene._reset = Mock()
    scene._recorder_manager = Mock()
    checkpoint = object()
    envs_idx = [1, 3]

    scene.restore(checkpoint, envs_idx=envs_idx)

    scene._reset.assert_called_once_with(checkpoint, envs_idx=envs_idx, keep_init=True)
    scene._recorder_manager.reset.assert_called_once_with(envs_idx)


def test_simulator_reset_restores_native_ipc_checkpoint_state():
    simulator = object.__new__(Simulator)
    simulator._solvers = []
    simulator._active_solvers = []
    simulator._queried_states = Mock()
    simulator._sensor_manager = Mock()
    simulator._coupler = Mock(spec=IPCCoupler)
    coupler_state = object()
    checkpoint = SimState(scene=None, s_global=0, f_local=0, solvers=[], coupler_state=coupler_state)

    simulator.reset(checkpoint)

    simulator._coupler.set_state.assert_called_once_with(coupler_state, envs_idx=None)
    simulator._coupler.reset.assert_not_called()
