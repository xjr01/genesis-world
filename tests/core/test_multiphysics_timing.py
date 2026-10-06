from dataclasses import dataclass

import pytest

from examples.multiphysics import ScenarioFrameRunner, ScenarioFrameState


@dataclass(frozen=True)
class Runtime:
    control_dt: float


@dataclass
class Controller:
    steps: int = 0

    def step(self, runtime: Runtime):
        self.steps += 1


@pytest.mark.parametrize(
    ("control_dt", "expected_steps"),
    (
        (0.002, 500),
        (1.0 / 60.0, 60),
        (1.0 / 120.0, 120),
        (0.02, 50),
        (3.5e-5, 28571),
    ),
)
def test_sixty_hertz_frames_distribute_native_substeps_without_drift(control_dt, expected_steps):
    controller = Controller()
    runner = ScenarioFrameRunner(controller, Runtime(control_dt))

    substeps = [runner.step() for _ in range(60)]

    assert sum(substeps) == expected_steps
    assert controller.steps == expected_steps
    assert max(substeps) - min(substeps) <= 1


def test_frame_clock_state_restores_fractional_substep_phase():
    runtime = Runtime(0.002)
    runner = ScenarioFrameRunner(Controller(), runtime)
    first_counts = [runner.step() for _ in range(17)]
    state = runner.get_state()
    expected_counts = [runner.step() for _ in range(13)]

    resumed = ScenarioFrameRunner(Controller(), runtime)
    resumed.set_state(state)

    assert sum(first_counts) == state.native_step_index
    assert [resumed.step() for _ in range(13)] == expected_counts


def test_frame_clock_rejects_inconsistent_checkpoint():
    runner = ScenarioFrameRunner(Controller(), Runtime(0.002))

    with pytest.raises(ValueError, match="inconsistent"):
        runner.set_state(ScenarioFrameState(frame_index=1, native_step_index=9))


def test_frame_clock_reset_restarts_substep_phase():
    runner = ScenarioFrameRunner(Controller(), Runtime(0.002))
    first_count = runner.step()
    runner.step()

    runner.reset()

    assert runner.step() == first_count
