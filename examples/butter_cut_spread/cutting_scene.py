"""Restrict compatible particle-in-cell (CPIC) separation to explicitly selected tools."""

from dataclasses import replace

import numpy as np

import genesis as gs
from genesis.engine.couplers.legacy_coupler import LegacyCoupler
from genesis.utils import array_class
from genesis.utils.misc import qd_to_numpy

qd = gs.qd


class CuttingScene(gs.Scene):
    """Scene-local coupling adapter; ordinary rigid contact remains enabled on supports.

    Call ``build(cutting_tools=(blade,), ...)`` after adding entities. Selection is
    immutable after build. This adapter uses the repository's native legacy coupler
    and its normal-side CPIC criterion; it does not replace the material or MPM kernels.
    """

    @gs.assert_unbuilt
    def build(self, *, cutting_tools, **kwargs):
        if not isinstance(self.sim.coupler, LegacyCoupler):
            raise TypeError("Tool selection requires the legacy coupler.")
        if not self.mpm_options.enable_CPIC:
            raise ValueError("Tool selection requires enable_CPIC=True.")
        tools = tuple(cutting_tools)
        if not tools:
            raise ValueError("Select at least one cutting tool.")
        for entity in tools:
            if entity not in self.rigid_solver.entities or not all(geom.needs_coup for geom in entity.geoms):
                raise ValueError("Cutting tools must be coupled rigid entities in this scene.")
        # Install before solver build captures the coupler. No process-wide classes
        # or coupling flags on the original rigid geometry metadata are changed.
        self.sim._coupler = ToolCoupler(self.sim, self.coupler_options, tools)
        return super().build(**kwargs)

    @gs.assert_built
    def get_cpic_transfer_counts(self):
        """Count separated transfers by rigid geometry in the last substep, summed over environments."""
        coupler = self.sim.coupler
        coupler.separation_counts.fill(0)
        count_separations(coupler.cpic_flag, coupler.separation_counts)
        return qd_to_numpy(coupler.separation_counts, transpose=True, copy=True)


class ToolCoupler(LegacyCoupler):
    def __init__(self, simulator, options, tools):
        super().__init__(simulator, options)
        self.tools = tools
        self.cutting_geoms_info = None
        self.separation_counts = None

    def build(self):
        super().build()
        if not self._rigid_mpm or not self.mpm_solver.enable_CPIC:
            raise ValueError("Tool selection requires active rigid-MPM coupling and CPIC.")
        selected = np.zeros(self.rigid_solver.n_geoms_, dtype=gs.np_bool)
        for entity in self.tools:
            for geom in entity.geoms:
                selected[geom.idx] = True
        mask = array_class.V(dtype=gs.qd_bool, shape=selected.shape)
        mask.from_numpy(selected)
        self.cutting_geoms_info = replace(self.rigid_solver.dyn_info.geoms, needs_coup=mask)
        self.separation_counts = qd.field(gs.qd_int, shape=self.rigid_solver.n_geoms)

    def mpm_surface_to_particle(self, f, geoms_state, geoms_info, sdf_info, rigid_info, collider_static_config):
        # Only this preprocessing call sees the mask. Unselected geometry normals
        # stay at their reset value zero, so their dot product cannot mark a transfer
        # as separated. Native grid contact still reads the original metadata.
        super().mpm_surface_to_particle(
            f, geoms_state, self.cutting_geoms_info, sdf_info, rigid_info, collider_static_config
        )


@qd.kernel
def count_separations(flags: qd.template(), counts: qd.template()):
    for index in qd.grouped(flags):
        geom_index = flags[index]
        if geom_index >= 0:
            qd.atomic_add(counts[geom_index], 1)
