"""Wall-clock accounting for offline rendering stages."""

from dataclasses import asdict, dataclass
from time import perf_counter


@dataclass
class Stage:
    seconds: float = 0.0
    calls: int = 0


class StageTimings:
    def __init__(self):
        self.stages: dict[str, Stage] = {}
        self.previous = perf_counter()

    def mark(self, name):
        now = perf_counter()
        elapsed = now - self.previous
        self.previous = now
        stage = self.stages.setdefault(name, Stage())
        stage.seconds += elapsed
        stage.calls += 1
        return elapsed

    def report(self):
        return {name: asdict(stage) for name, stage in self.stages.items()}
