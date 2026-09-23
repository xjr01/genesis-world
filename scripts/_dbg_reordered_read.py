import sys

sys.path.insert(0, r"D:/workspace/python-workspace/ipbf/multiflow/genesis-world")
import numpy as np
import quadrants as qd
import genesis as gs
from genesis.utils.misc import qd_to_numpy

gs.init(backend=gs.gpu, precision="32", seed=0)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1 / 240, substeps=4, gravity=(0, 0, 0)),
    pbstf_options=gs.options.PBSTFOptions(
        particle_size=0.002,
        lower_bound=(-0.06, -0.06, -0.01),
        upper_bound=(0.06, 0.06, 0.11),
        max_solver_iterations=20,
        topology_rebuild_interval=2,
    ),
    show_viewer=False,
)
scene.add_entity(
    material=gs.materials.PBSTF.Liquid(rho=1000.0, surface_tension_compliance=1e12, sampler="regular"),
    morph=gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0, 0, 0.05)),
)
scene.build()
solver = scene.sim.pbstf_solver


@qd.kernel
def dump_pos(out: qd.types.ndarray()):
    for i in range(3375):
        for a in qd.static(range(3)):
            out[i, a] = solver.particles_reordered[i, 0].pos[a]


out = np.zeros((3375, 3), dtype=np.float32)
dump_pos(out)
print("GPU kernel dump bounds:", out.min(0), out.max(0))
np_read = qd_to_numpy(solver.particles_reordered.pos, transpose=True)[0]
print("numpy read bounds:    ", np_read.min(0), np_read.max(0))
print("kernel row0:", out[0], " numpy row0:", np_read[0])
print("identical:", np.array_equal(out, np_read))
