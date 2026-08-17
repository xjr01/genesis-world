"""IPBF example 06: large-scale block flop (paper Fig. 12).

A 0.5x0.8x0.5 m liquid block at particle_size=0.008 (384,400 fluid particles + ~94k boundary,
> 250k fluid as in the paper's Block Flop) collapses in the 1x1x1 m box.
Config: h = 1/480 s, alpha=0, damping off (default), boundary particles on (default).
Two passes in one process: timing without recording, then video.
Paper reference: Fig. 12 / Fig. 1 (Diaz et al., SIGGRAPH Asia 2025); their numbers (RTX 4090):
250k particles @1 iteration ~50 ms/frame, 1M particles 159 ms/frame.
Run: python examples/ipbf/06_large_scale.py [--iters 1] [--horizon 200]
"""

import argparse
import os
import time

import numpy as np
import torch

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def build_scene(with_camera: bool, iters: int, particle_size: float):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60, substeps=8),
        ipbf_options=gs.options.IPBFOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            particle_size=particle_size,
            ipbf_iterations=iters,
            alpha=0.0,
        ),
        show_viewer=False,
    )
    liquid = scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(-0.25, 0.0, 0.25), size=(0.5, 0.8, 0.5)),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
    )
    cam = None
    if with_camera:
        cam = scene.add_camera(res=(1280, 960), pos=(1.8, -1.8, 1.2), lookat=(0, 0, 0.3), fov=35, GUI=False)
    scene.build()
    return scene, liquid, cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--particle-size", type=float, default=0.008)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "large_scale.mp4"))
    parser.add_argument("--metrics", default=os.path.join(OUTPUT_DIR, "large_scale_metrics.txt"))
    args = parser.parse_args()

    test_mode = "PYTEST_VERSION" in os.environ  # quick CPU smoke for tests/test_examples.py
    backend = args.backend or ("cpu" if test_mode else "gpu")
    if test_mode:
        args.horizon = min(args.horizon, 2)
        args.warmup = min(args.warmup, 1)

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

    # ---- pass A: timing (no recording) ----
    scene, liquid, _ = build_scene(with_camera=False, iters=args.iters, particle_size=args.particle_size)
    solver = scene.sim.ipbf_solver
    n_fluid, n_boundary = solver._n_fluid_particles, solver._n_boundary_particles
    print(f"pass A (timing): {n_fluid} fluid + {n_boundary} boundary particles, iters={args.iters}, damping off")

    step_times = []
    last_avg_C, last_max_C = float("nan"), float("nan")
    for frame in range(args.horizon):
        t0 = time.perf_counter()
        scene.step()
        step_times.append(time.perf_counter() - t0)
        if frame == args.horizon - 1:
            rho = liquid.get_particles_rho()
            assert torch.isfinite(rho).all(), "Non-finite rho!"
            C = torch.clamp(rho - 1.0, min=0.0)
            last_avg_C, last_max_C = C.mean().item(), C.max().item()
    del scene

    timed = np.array(step_times[args.warmup :]) * 1e3
    ms_mean, ms_std = timed.mean(), timed.std()

    # ---- pass B: video ----
    scene, liquid, cam = build_scene(with_camera=True, iters=args.iters, particle_size=args.particle_size)
    cam.start_recording(save_to_filename=args.out, fps=60)
    for _ in range(args.horizon):
        scene.step()
    cam.stop_recording()
    pos = liquid.get_particles_pos()
    assert torch.isfinite(pos).all(), "Non-finite positions (video pass)!"
    del scene

    lines = [
        "IPBF large-scale benchmark",
        f"scene: 0.5x0.8x0.5 m block flop in 1x1x1 m box, ps={args.particle_size}",
        f"particles: {n_fluid} fluid + {n_boundary} boundary = {n_fluid + n_boundary}",
        f"h = 1/480 s (dt=1/60, substeps=8), ipbf_iterations = {args.iters}, alpha = 0, damping = off",
        f"boundary particles: on; hardware: {torch.cuda.get_device_name(0) if backend == 'gpu' else 'CPU'}",
        f"ms/frame: {ms_mean:.2f} +- {ms_std:.2f} (mean over {len(timed)} steps, {args.warmup} warmup excluded)",
        f"final-frame avg_C {last_avg_C:.6e}, max_C {last_max_C:.6e}",
        "paper reference (RTX 4090): 250k particles @1 iteration ~50 ms/frame; 1M particles 159 ms/frame",
    ]
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "w") as fp:
        fp.write(text + "\n")
    print("metrics saved to", args.metrics, "; video saved to", args.out)


if __name__ == "__main__":
    main()
