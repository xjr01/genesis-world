"""Limit display notification overhead to the butter task's rendering needs."""

from contextlib import contextmanager


@contextmanager
def physics_only_visual_updates(scene):
    """Suspend this scene's display subscriptions during a loop with no visual-state queries."""
    if scene.viewer is not None or scene.visualizer.cameras:
        yield
        return

    suspended = []
    # Solver exposes registration but no removal API; retain each exact subscription for restoration.
    for solver in (scene.rigid_solver, scene.kinematic_solver, scene.pbd_solver):
        for subscriber in scene.visualizer._render_state_subscribers:
            if subscriber in solver._subscribers:
                suspended.append((solver, subscriber))
                solver._subscribers.remove(subscriber)
    try:
        yield
    finally:
        for solver, subscriber in suspended:
            solver.subscribe(subscriber)
        scene.visualizer.reset()
