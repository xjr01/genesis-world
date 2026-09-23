# MF-18 phase A fidelity check: run the identical PBSTF scene on two engines in
# separate processes and compare public state step by step.
#   root engine:       PYTHONIOENCODING=utf-8 "$PY" _dbg_pbstf_port_compare.py root
#   multiflow engine:  PYTHONIOENCODING=utf-8 "$PY" _dbg_pbstf_port_compare.py multiflow
import sys

ENGINE = sys.argv[1]
if ENGINE == "multiflow":
    sys.path.insert(0, r"D:/workspace/python-workspace/ipbf/multiflow/genesis-world")

import numpy as np

import genesis as gs

if ENGINE == "multiflow":
    assert "multiflow" in gs.__file__, gs.__file__
else:
    assert "multiflow" not in gs.__file__, gs.__file__

from genesis.utils.misc import qd_to_numpy, tensor_to_array

DT = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2

gs.init(backend=gs.gpu, precision="32", seed=0)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT, substeps=1),
    pbstf_options=gs.options.PBSTFOptions(
        particle_size=0.002,
        lower_bound=(-0.1, -0.1, -0.01),
        upper_bound=(0.1, 0.1, 0.1),
    ),
    show_viewer=False,
)
scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.02),
        size=(0.02, 0.02, 0.02),
    ),
    material=gs.materials.PBSTF.Liquid(rho=1000.0, sampler="regular"),
)
scene.build()

sim = scene.sim
solver = sim.pbstf_solver
mass = solver._default_mass

print(f"[{ENGINE}] dt = {sim._dt:.10e}  substeps = {sim._substeps}")
print(f"[{ENGINE}] support_radius = {solver._support_radius:.10e}")
print(f"[{ENGINE}] gravity = {sim._gravity}")
for name in (
    "_max_solver_iterations",
    "_max_surface_tension_solver_iterations",
    "_max_viscosity_solver_iterations",
):
    if hasattr(solver, name):
        print(f"[{ENGINE}] {name} = {getattr(solver, name)}")


def _snap():
    state = solver.get_state(0)
    return np.concatenate(
        (tensor_to_array(state.pos).reshape(-1), tensor_to_array(state.vel).reshape(-1))
    )


snapshots = [_snap()]  # step 0: right after build, before any scene.step()
for i in range(5):
    scene.step()
    snapshots.append(_snap())

density = qd_to_numpy(solver.particles.density, transpose=True)[0]
out = f"_pbstf_port_compare_{ENGINE}_dt{DT:g}.npz"
np.savez(out, mass=mass, density=density, snapshots=np.stack(snapshots))
print(f"[{ENGINE}] mass = {mass:.10e}")
print(f"[{ENGINE}] density ratio mean = {(density / 1000.0).mean():.8f}")
print(f"[{ENGINE}] saved {out}")
