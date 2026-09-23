"""IPBF example 05: Double Dam Break benchmark (paper Table 1).

Two water columns on opposite sides of the box collapse and collide in the middle, forming
a thin jet. Config: box 1x1x1 m, particle_size=0.01, h = 1/480 s, ipbf_iterations=2, alpha=0,
damping off (default), boundary particles on (default).
Two passes in one process: timing without recording (ms/frame of scene.step, warmup excluded,
density error sampled), then a fresh scene records the video.
Paper reference: Table 1 / Figs. 2-3 (Diaz et al., SIGGRAPH Asia 2025); their numbers (RTX 4090,
particle count unknown): avg density error 9.2e-4, 70 ms/frame.
Run: python examples/ipbf/05_double_dam_break.py [--horizon 200] [--iters 2]
"""

import argparse
import os
import time

import numpy as np
import torch

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def build_scene(with_camera: bool, iters: int):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60, substeps=8),
        ipbf_options=gs.options.IPBFOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            particle_size=0.01,
            ipbf_iterations=iters,
            alpha=0.0,
        ),
        show_viewer=False,
    )
    left = scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(-0.3, 0.0, 0.3), size=(0.3, 0.8, 0.6)),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
    )
    scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(0.3, 0.0, 0.3), size=(0.3, 0.8, 0.6)),
        surface=gs.surfaces.Default(color=(1.0, 0.6, 0.3), vis_mode="particle"),
    )
    cam = None
    if with_camera:
        cam = scene.add_camera(res=(1280, 960), pos=(0.0, -2.2, 1.3), lookat=(0, 0, 0.35), fov=35, GUI=False)
    scene.build()
    return scene, left, cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "double_dam_break.mp4"))
    parser.add_argument("--metrics", default=os.path.join(OUTPUT_DIR, "ddb_metrics.txt"))
    args = parser.parse_args()

    test_mode = "PYTEST_VERSION" in os.environ  # quick CPU smoke for tests/test_examples.py
    backend = args.backend or ("cpu" if test_mode else "gpu")
    if test_mode:
        args.horizon = min(args.horizon, 6)
        args.warmup = min(args.warmup, 2)

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

    # ---- pass A: timing + density error (no recording) ----
    scene, liquid, _ = build_scene(with_camera=False, iters=args.iters)
    solver = scene.sim.ipbf_solver
    n_fluid, n_boundary = solver._n_fluid_particles, solver._n_boundary_particles
    print(f"pass A (timing): {n_fluid} fluid + {n_boundary} boundary particles, iters={args.iters}, damping off")

    step_times = []
    err_samples = []
    for frame in range(args.horizon):
        t0 = time.perf_counter()
        scene.step()
        step_times.append(time.perf_counter() - t0)
        if frame >= args.horizon // 2 and frame % 5 == 0:
            rho = liquid.get_particles_rho()
            assert torch.isfinite(rho).all(), f"Non-finite rho at frame {frame}!"
            C = torch.clamp(rho - 1.0, min=0.0)
            err_samples.append((frame, C.mean().item(), C.max().item()))
    del scene

    timed = np.array(step_times[args.warmup :]) * 1e3
    ms_mean, ms_std = timed.mean(), timed.std()
    avg_C = float(np.mean([e[1] for e in err_samples]))
    max_C = float(np.max([e[2] for e in err_samples]))

    # ---- pass B: video ----
    scene, liquid, cam = build_scene(with_camera=True, iters=args.iters)
    cam.start_recording(save_to_filename=args.out, fps=60)
    for _ in range(args.horizon):
        scene.step()
    cam.stop_recording()
    pos = liquid.get_particles_pos()
    assert torch.isfinite(pos).all(), "Non-finite positions (video pass)!"
    del scene

    lines = [
        "IPBF Double Dam Break benchmark",
        f"scene: two 0.3x0.8x0.6 columns in 1x1x1 m box, ps=0.01",
        f"particles: {n_fluid} fluid + {n_boundary} boundary = {n_fluid + n_boundary}",
        f"h = 1/480 s (dt=1/60, substeps=8), ipbf_iterations = {args.iters}, alpha = 0, damping = off",
        f"boundary particles: on; hardware: {torch.cuda.get_device_name(0) if backend == 'gpu' else 'CPU'}",
        f"ms/frame: {ms_mean:.2f} +- {ms_std:.2f} (mean over {len(timed)} steps, {args.warmup} warmup excluded)",
        f"avg density error avg_C (frames {args.horizon // 2}..{args.horizon - 1}): {avg_C:.6e}",
        f"max density error max_C (same window): {max_C:.6e}",
        "paper reference (RTX 4090, their particle count unknown): avg err 9.2e-4, 70 ms/frame",
    ]
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "w") as fp:
        fp.write(text + "\n")
    print("metrics saved to", args.metrics, "; video saved to", args.out)


if __name__ == "__main__":
    main()
