from dataclasses import dataclass

from examples.multiflow.teapot.pbstf_surface_tension import update_mop_case

from .config import TableWipingTaskConfig
from .scene import TableWipingRuntime


@dataclass
class TableWipingController:
    config: TableWipingTaskConfig
    qpos: object | None = None

    def reset(self, runtime: TableWipingRuntime):
        self.qpos = runtime.initial_qpos

    def step(self, runtime: TableWipingRuntime):
        if self.qpos is None:
            self.reset(runtime)
        update = update_mop_case(runtime.scene, runtime.scene.cur_t, runtime.settings, self.qpos)
        self.qpos = update.qpos
        runtime.scene.step()
