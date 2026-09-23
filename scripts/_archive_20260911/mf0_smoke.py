"""MF-0 smoke test: stock PBF liquid box free-fall on the multiflow genesis copy.

Verifies: genesis import resolves to the multiflow copy, GPU backend,
PBD liquid pipeline, particle rendering + video recording.
Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" multiflow/scripts/mf0_smoke.py
"""

import argparse
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import genesis as gs

print("genesis.__file__ =", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), (
    "genesis must come from the multiflow copy, got " + gs.__file__
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--out", default=os.path.join(WORKSPACE, "videos", "mf0_smoke.mp4"))
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32")

    dt = 2e-3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=5),
        pbd_options=gs.options.PBDOptions(
            lower_bound=(-0.3, -0.3, 0.0),
            upper_bound=(0.3, 0.3, 0.8),
            particle_size=0.01,
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
        ),
        show_viewer=False,
    )

    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Box(lower=(-0.1, -0.1, 0.3), upper=(0.1, 0.1, 0.5)),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
    )

    cam = scene.add_camera(res=(960, 720), pos=(1.0, -1.0, 0.6), lookat=(0.0, 0.0, 0.25), fov=40, GUI=False)
    scene.build()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_steps = int(args.seconds / dt)
    cam.start_recording(save_to_filename=args.out, fps=60)
    for _ in range(n_steps):
        scene.step()
    cam.stop_recording()

    pos = liquid.get_particles_pos()
    print("n_particles:", liquid.n_particles)
    print("final pos stats: min", pos.min(axis=0), "max", pos.max(axis=0))
    print("video saved to", args.out)


if __name__ == "__main__":
    main()
