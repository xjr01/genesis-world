# MF-18 phase A smoke test: PBSTF port into the multiflow copy + PBD zero-regression.
# Run with: PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" _dbg_pbstf_port_smoke.py
import sys

sys.path.insert(0, r"D:/workspace/python-workspace/ipbf/multiflow/genesis-world")

import numpy as np

import genesis as gs

assert "multiflow" in gs.__file__, gs.__file__

from genesis.utils.misc import qd_to_numpy


def run_pbstf_smoke():
    scene = gs.Scene(
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=0.002,
            lower_bound=(-0.1, -0.1, -0.01),
            upper_bound=(0.1, 0.1, 0.1),
        ),
        show_viewer=False,
    )
    fluid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.02),
            size=(0.02, 0.02, 0.02),
        ),
        material=gs.materials.PBSTF.Liquid(rho=1000.0, sampler="regular"),
    )
    scene.build()

    solver = scene.sim.pbstf_solver
    n_particles = solver.n_particles
    print(f"[pbstf] n_particles = {n_particles}")
    assert n_particles == 1000, n_particles
    assert fluid.n_particles == 1000

    # Mass calibration: regular lattice, ps=0.002, support=0.006 -> sum W = 125224338.17 m^-3
    # (MF-18 plan section 2), so m = rho0 / max_density ~= 7.985668e-6 kg.
    default_mass = solver._default_mass
    expected_mass = 1000.0 / 125224338.17
    mass_dev = abs(default_mass - expected_mass) / expected_mass
    print(f"[pbstf] calibrated default_mass = {default_mass:.6e} kg (expected ~= {expected_mass:.6e}, dev {mass_dev:.4%})")

    for i in range(5):
        scene.step()

    state = solver.get_state(0)
    pos = state.pos
    vel = state.vel
    is_finite = bool(np.isfinite(qd_to_numpy(solver.particles.pos, transpose=True)).all())
    print(f"[pbstf] state pos shape = {tuple(pos.shape)}, finite(pos+vel) = {bool(pos.isfinite().all() and vel.isfinite().all())}, field finite = {is_finite}")
    assert bool(pos.isfinite().all()) and bool(vel.isfinite().all())
    assert is_finite

    density = qd_to_numpy(solver.particles.density, transpose=True)[0]
    density_ratio = density / 1000.0
    print(
        f"[pbstf] density ratio mean = {density_ratio.mean():.6f}, "
        f"min = {density_ratio.min():.6f}, max = {density_ratio.max():.6f}"
    )
    return mass_dev


def run_pbd_regression():
    scene = gs.Scene(
        pbd_options=gs.options.PBDOptions(
            particle_size=0.01,
            lower_bound=(-0.5, -0.5, -0.01),
            upper_bound=(0.5, 0.5, 0.5),
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.04, 0.04, 0.04),
        ),
        material=gs.materials.PBD.Liquid(),
    )
    scene.build()
    scene.step()
    scene.step()
    pos = qd_to_numpy(scene.sim.pbd_solver.particles.pos, transpose=True)
    print(f"[pbd] regression scene OK, n_particles = {scene.sim.pbd_solver.n_particles}, finite = {bool(np.isfinite(pos).all())}")
    assert np.isfinite(pos).all()


if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    mass_dev = run_pbstf_smoke()
    run_pbd_regression()
    if mass_dev > 0.01:
        print(f"[pbstf] WARNING: mass calibration deviates by {mass_dev:.4%} (> 1%)")
    print("SMOKE_OK")
