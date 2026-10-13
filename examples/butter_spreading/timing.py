"""Accumulate wall-clock time for repeated processing stages."""

from dataclasses import dataclass


@dataclass
class StageTiming:
    seconds: float = 0.0
    calls: int = 0


class StageTimings:
    def __init__(self):
        self.stages: dict[str, StageTiming] = {}

    def add(self, name: str, seconds: float):
        stage = self.stages.setdefault(name, StageTiming())
        stage.seconds += seconds
        stage.calls += 1
