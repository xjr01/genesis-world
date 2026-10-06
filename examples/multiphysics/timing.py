import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, Protocol, TypeVar


class ScenarioRuntime(Protocol):
    @property
    def control_dt(self) -> float:
        """Duration advanced by one native controller step."""


RuntimeT = TypeVar("RuntimeT", bound=ScenarioRuntime)


class ScenarioController(Protocol[RuntimeT]):
    def step(self, runtime: RuntimeT) -> object:
        """Advance one native controller step."""


@dataclass(frozen=True)
class ScenarioFrameState:
    """Clock state required to resume a fixed-rate adapter stream."""

    frame_index: int
    native_step_index: int


class ScenarioFrameRunner(Generic[RuntimeT]):
    """Advance a scenario at a fixed adapter frame rate using native controller substeps.

    The controller keeps its calibrated time step. Each adapter frame executes the number of complete native steps
    whose simulated timestamps fall within that frame, distributing fractional ratios without long-term drift.
    """

    def __init__(
        self,
        controller: ScenarioController[RuntimeT],
        runtime: RuntimeT,
        *,
        fps: float = 60.0,
    ) -> None:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(runtime.control_dt) or runtime.control_dt <= 0.0:
            raise ValueError("runtime.control_dt must be finite and positive")
        self.controller = controller
        self.runtime = runtime
        self.fps = fps
        self._native_steps_per_frame = Fraction(1, 1) / Fraction(str(fps)) / Fraction(str(runtime.control_dt))
        self.frame_index = 0
        self.native_step_index = 0

    def get_state(self) -> ScenarioFrameState:
        return ScenarioFrameState(self.frame_index, self.native_step_index)

    def reset(self) -> None:
        self.frame_index = 0
        self.native_step_index = 0

    def set_state(self, state: ScenarioFrameState) -> None:
        if state.frame_index < 0 or state.native_step_index < 0:
            raise ValueError("frame and native-step indices must be non-negative")
        expected_native_steps = int(state.frame_index * self._native_steps_per_frame)
        if state.native_step_index != expected_native_steps:
            raise ValueError(
                f"frame state is inconsistent: frame {state.frame_index} requires "
                f"{expected_native_steps} native steps, got {state.native_step_index}"
            )
        self.frame_index = state.frame_index
        self.native_step_index = state.native_step_index

    def substeps_for_next_frame(self) -> int:
        target_native_steps = int((self.frame_index + 1) * self._native_steps_per_frame)
        return target_native_steps - self.native_step_index

    def step(self) -> int:
        """Advance one adapter frame and return the executed native substep count."""
        substeps = self.substeps_for_next_frame()
        for _ in range(substeps):
            self.controller.step(self.runtime)
        self.native_step_index += substeps
        self.frame_index += 1
        return substeps
