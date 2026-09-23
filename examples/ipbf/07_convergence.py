"""IPBF example 07: density-error vs iteration-count convergence curve (paper Fig. 10).

Method: evolve the dam-break scene at h = 1/480 s with iters=2; snapshot the solver state at
two representative frames (mid-splash and settling). From each snapshot, restore the exact
same state and run ONE full step (8 substeps) with ipbf_iterations forced to 1/2/4/8, then
measure avg/max of the clamped constraint C = max(rho~ - 1, 0). The iteration count is a
host-side loop bound (IPBFSolver._ipbf_iterations), so it can be varied without recompiling.
Damping off (default), boundary particles on (default).
Paper reference: Fig. 10 (Diaz et al., SIGGRAPH Asia 2025) — IPBF converges to low density
error in <10 iterations.
Run: python examples/ipbf/07_convergence.py
"""

import argparse
import csv
import os

import torch

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

SNAPSHOT_FRAMES = [30, 100]
ITER_COUNTS = [1, 2, 4, 8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "convergence.csv"))
    args = parser.parse_args()

    test_mode = "PYTEST_VERSION" in os.environ  # quick CPU smoke for tests/test_examples.py
    backend = args.backend or ("cpu" if test_mode else "gpu")
    snapshot_frames = [1, 2] if test_mode else SNAPSHOT_FRAMES

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60, substeps=8),
        ipbf_options=gs.options.IPBFOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            particle_size=0.01,
            ipbf_iterations=2,
            alpha=0.0,
        ),
        show_viewer=False,
    )
    liquid = scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(-0.25, 0.0, 0.25), size=(0.4, 0.8, 0.5)),
        surface=gs.surfaces.Default(vis_mode="particle"),
    )
    scene.build()
    solver = scene.sim.ipbf_solver

    # evolve and snapshot
    snapshots = {}
    target = max(snapshot_frames)
    for frame in range(target + 1):
        if frame in snapshot_frames:
            snapshots[frame] = solver.get_state(0)
        scene.step()

    # from each snapshot: same state, one full step, varying iteration count
    rows = []
    for frame_tag, state in snapshots.items():
        for k in ITER_COUNTS:
            solver.set_state(0, state)
            solver._ipbf_iterations = k
            scene.step()
            rho = liquid.get_particles_rho()
            assert torch.isfinite(rho).all(), f"Non-finite rho (frame_tag={frame_tag}, iters={k})"
            C = torch.clamp(rho - 1.0, min=0.0)
            rows.append((frame_tag, k, C.mean().item(), C.max().item()))
            print(f"snapshot f{frame_tag}, iters={k}: avg_C {rows[-1][2]:.6e}, max_C {rows[-1][3]:.6e}")
    solver._ipbf_iterations = 2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["snapshot_frame", "iters", "avg_C", "max_C"])
        writer.writerows(rows)
    print("convergence log saved to", args.out)


if __name__ == "__main__":
    main()
