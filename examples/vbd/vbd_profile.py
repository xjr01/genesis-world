"""Profile the remeshed jelly drop with wall timers, Python tracing, or device kernel timers."""

import argparse
import cProfile
import hashlib
import inspect
import json
import logging
import os
import platform
import pstats
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import tetgen
import trimesh

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.vbd.solver import VBDSolver
from genesis.utils.misc import tensor_to_array


@dataclass
class FrameTiming:
    repeat: int
    frame: int
    milliseconds: float


@dataclass
class KernelTiming:
    name: str
    device_tasks: int
    total_ms: float
    mean_ms: float


@dataclass
class LineTiming:
    line: int
    source: str
    hits: int
    host_ms: float


@dataclass
class SubstepCounts:
    iterations: int
    final_candidate_pairs: int


@dataclass
class ProfileReport:
    mode: str
    revision: str
    genesis_path: str
    platform: str
    python: str
    torch: str
    device: str
    asset: str
    asset_sha256: str
    vertices: int
    tetrahedra: int
    surface_triangles: int
    threads_per_vertex: int
    color_sizes: list[int]
    frame_dt: float
    substeps: int
    iteration_limit: int
    convergence_tolerance: float
    precision: str
    material_model: str
    youngs_modulus: float
    poisson_ratio: float
    density: float
    friction: float
    contact_distance: float
    is_line_search_enabled: bool
    line_search_iterations: int
    warmup_frames: int
    measured_frames_per_repeat: int
    repeats: int
    mean_frame_ms: float
    median_frame_ms: float
    p95_frame_ms: float
    total_device_ms: float | None
    final_minimum_height: float
    final_minimum_volume_ratio: float
    timings: list[FrameTiming]
    kernels: list[KernelTiming]
    lines: list[LineTiming]
    substep_counts: list[SubstepCounts]


class SolverLineTimer:
    """Attribute host elapsed time to solver lines; synchronization includes queued device work."""

    def __init__(self):
        self.code = VBDSolver.substep_pre_coupling.__code__
        self.source, self.first_line = inspect.getsourcelines(VBDSolver.substep_pre_coupling)
        self.nanoseconds = [0] * len(self.source)
        self.hits = [0] * len(self.source)
        self.previous_line = None
        self.previous_tick = 0
        self.substeps = []

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        tick = time.perf_counter_ns()
        if event == "call":
            self.previous_line = None
        if self.previous_line is not None:
            self.nanoseconds[self.previous_line] += tick - self.previous_tick
        if event == "line":
            self.previous_line = frame.f_lineno - self.first_line
            self.hits[self.previous_line] += 1
        elif event == "return":
            self.substeps.append(SubstepCounts(frame.f_locals["iteration"] + 1, frame.f_locals["n_contacts"]))
            self.previous_line = None
        self.previous_tick = time.perf_counter_ns()
        return self.trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "kernels", "python", "lines"), default="baseline")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--line-search",
        dest="is_line_search_enabled",
        action="store_true",
        help="Enable local energy backtracking at extra cost.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, default=Path("out/vbd/profile"))
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.repeats, args.iterations) <= 0:
        parser.error("Step, warmup, repeat and iteration counts must be positive.")
    if args.mode == "kernels" and args.repeats != 1:
        parser.error("Kernel timing uses one reset run so its records cover only measured steps.")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    ).strip()
    os.environ["QD_KERNEL_PROFILER"] = "1" if args.mode == "kernels" else "0"
    gs.init(backend=gs.cuda, precision="32", logging_level=logging.WARNING)
    asset = Path(gs.__file__).parent / "assets" / "meshes" / "jelly" / "remeshed.obj"
    mesh = trimesh.load(asset, force="mesh", process=False)
    mesh.vertices[:, 2] += 0.015 - mesh.vertices[:, 2].min()
    tetrahedralizer = tetgen.TetGen(mesh.vertices, mesh.faces.astype(np.int32))
    vertices, elements, *_ = tetrahedralizer.tetrahedralize(switches="pq1.5/10Ya0.000000100000Q")
    if not np.allclose(vertices[: len(mesh.vertices)], mesh.vertices):
        raise ValueError("Tetrahedralization must preserve the surface mapping.")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            substeps=2,
            gravity=(0, 0, -9.81),
        ),
        vbd_options=gs.options.VBDOptions(
            iterations=args.iterations,
            convergence_tolerance=args.tolerance,
            contact_d_hat=0.0005,
            is_line_search_enabled=args.is_line_search_enabled,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.22, -0.28, 0.2),
            camera_lookat=(0.0, 0.0, 0.035),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.VBD.Elastic(E=10000.0, nu=0.25, rho=360.0, friction_mu=0.4),
    )
    scene.build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(args.warmup):
        scene.step()
    torch.cuda.synchronize()
    print(f"Warmup complete: {args.warmup} frames", flush=True)
    timings, kernels, lines = [], [], []
    line_timer = SolverLineTimer()
    python_profiler = cProfile.Profile()
    total_device_ms = None
    for repeat in range(args.repeats):
        scene.reset()
        torch.cuda.synchronize()
        if args.mode == "kernels" and repeat == 0:
            qd.profiler.clear_kernel_profiler_info()
        if args.mode == "python":
            python_profiler.enable()
        elif args.mode == "lines":
            sys.settrace(line_timer.trace)
        try:
            for step in range(args.steps):
                tick = time.perf_counter()
                scene.step()
                torch.cuda.synchronize()
                timings.append(FrameTiming(repeat + 1, step + 1, (time.perf_counter() - tick) * 1000))
        finally:
            python_profiler.disable()
            sys.settrace(None)
        print(
            f"Repeat {repeat + 1}: {np.mean([row.milliseconds for row in timings[-args.steps :]]):.3f} ms/frame",
            flush=True,
        )
    stem = f"{args.mode}_i{args.iterations}_tol{args.tolerance:g}_ls{'on' if args.is_line_search_enabled else 'off'}"
    if args.mode == "python":
        python_profiler.dump_stats(args.output_dir / f"{stem}.pstats")
        with (args.output_dir / f"{stem}.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(python_profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(80)
            pstats.Stats(python_profiler, stream=stream).strip_dirs().sort_stats("tottime").print_stats(50)
    elif args.mode == "kernels":
        total_device_ms = qd.profiler.get_kernel_profiler_total_time() * 1000
        records = qd.lang.impl.get_runtime().prog.get_kernel_profiler_records()
        for name in (
            "solve_colors",
            "solve_color",
            "apply_color",
            "find_contacts",
            "kernel_refit_bvh",
            "refresh_contacts",
            "update_bounds",
            "update_translation",
            "predict",
            "prepare_prediction",
            "limit_prediction",
            "limit_initialization",
            "apply_prediction",
            "finish",
            "validate_state",
        ):
            durations = [record.kernel_time for record in records if record.name.startswith(name + "_")]
            if durations:
                kernels.append(KernelTiming(name, len(durations), sum(durations), np.mean(durations)))
        with (args.output_dir / f"{stem}.txt").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            qd.profiler.print_kernel_profiler_info()
    elif args.mode == "lines":
        for index, source in enumerate(line_timer.source):
            if line_timer.hits[index]:
                lines.append(
                    LineTiming(
                        line_timer.first_line + index,
                        source.strip(),
                        line_timer.hits[index],
                        line_timer.nanoseconds[index] / 1e6,
                    )
                )
    final_positions = tensor_to_array(entity.get_state().pos)[0]
    rest_edges = vertices[elements[:, 1:]] - vertices[elements[:, :1]]
    final_edges = final_positions[elements[:, 1:]] - final_positions[elements[:, :1]]
    volume_ratio = np.linalg.det(final_edges) / np.linalg.det(rest_edges)
    report = ProfileReport(
        args.mode,
        revision,
        gs.__file__,
        platform.platform(),
        sys.version,
        torch.__version__,
        torch.cuda.get_device_name(),
        asset.relative_to(Path(gs.__file__).parent / "assets").as_posix(),
        hashlib.sha256(asset.read_bytes()).hexdigest(),
        entity.n_vertices,
        entity.n_elements,
        len(mesh.faces),
        scene.vbd_solver.config.n_vertex_threads,
        np.diff(scene.vbd_solver.colors.offsets).tolist(),
        scene.dt,
        scene.sim.substeps,
        args.iterations,
        args.tolerance,
        "32",
        "log_neohookean",
        entity.material.E,
        entity.material.nu,
        entity.material.rho,
        entity.material.friction_mu,
        scene.vbd_options.contact_d_hat,
        scene.vbd_options.is_line_search_enabled,
        scene.vbd_options.line_search_iterations,
        args.warmup,
        args.steps,
        args.repeats,
        np.mean([row.milliseconds for row in timings]),
        np.median([row.milliseconds for row in timings]),
        np.percentile([row.milliseconds for row in timings], 95),
        total_device_ms,
        float(final_positions[:, 2].min()),
        float(volume_ratio.min()),
        timings,
        kernels,
        lines,
        line_timer.substeps,
    )
    report_path = args.output_dir / f"{stem}.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
