import pytest

import genesis as gs
from examples.butter_spreading.runtime import physics_only_visual_updates
from genesis.engine.solvers.base_solver import StateChange, Subscriber


@pytest.mark.parametrize("has_camera", [False, True])
def test_butter_visual_notifications_are_scene_local_and_restored(has_camera):
    scene = gs.Scene(show_viewer=False)
    blade = scene.add_entity(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), fixed=True))
    if has_camera:
        scene.add_camera(res=(64, 64), pos=(0.4, -0.5, 0.3), lookat=(0.0, 0.0, 0.0), GUI=False)
    scene.build()
    other_scene = gs.Scene(show_viewer=False)
    other_entity = other_scene.add_entity(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), fixed=True))
    other_scene.build()
    other_scene.visualizer.update_visual_states()
    observer = Subscriber(to=frozenset((StateChange.GEOMETRY,)))
    scene.rigid_solver.subscribe(observer)
    subscribers = scene.rigid_solver._subscribers.copy()
    other_subscribers = other_scene.rigid_solver._subscribers.copy()
    visual_subscribers = set(scene.visualizer._render_state_subscribers)
    assert visual_subscribers

    with physics_only_visual_updates(scene):
        assert scene.rigid_solver._subscribers == (subscribers if has_camera else subscribers - visual_subscribers)
        blade.set_pos((0.02, 0.0, 0.0), relative=False)
        assert StateChange.GEOMETRY in observer.pending
        other_entity.set_pos((0.03, 0.0, 0.0), relative=False)
        assert other_scene.visualizer._t == -1
        assert other_scene.rigid_solver._subscribers == other_subscribers
    assert scene.rigid_solver._subscribers == subscribers

    with pytest.raises(RuntimeError, match="stop the butter loop"), physics_only_visual_updates(scene):
        raise RuntimeError("stop the butter loop")
    assert scene.rigid_solver._subscribers == subscribers
    scene.visualizer.update_visual_states()
    assert scene.visualizer._t == scene._t
    blade.set_pos((0.04, 0.0, 0.0), relative=False)
    assert scene.visualizer._t == -1
