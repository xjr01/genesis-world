"""Profile the remeshed jelly shell or volume with wall, Python, or device kernel timers."""

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
    shell_rebuilds: int
    status_reads: int
    solver_submissions: int
    maximum_candidate_pairs: int


@dataclass
class ProfileReport:
    mode: str
    model: str
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
    contact_stiffness: float
    shell_contact_stiffness: float
    contact_search_radius: float
    thickness: float | None
    bending_rigidity: float | None
    shear_stiffness: float | None
    is_bending_enabled: bool
    is_contact_diagnostics_enabled: bool
    is_line_search_enabled: bool
    line_search_iterations: int
    setup_and_warmup_seconds: float
    warmup_frames: int
    measured_frames_per_repeat: int
    repeats: int
    mean_frame_ms: float
    median_frame_ms: float
    p95_frame_ms: float
    total_device_ms: float | None
    final_minimum_height: float
    final_minimum_volume_ratio: float | None
    final_minimum_area_ratio: float
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
        self.shell_rebuilds = 0
        self.status_reads = 0
        self.solver_submissions = 0
        self.maximum_candidate_pairs = 0

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        tick = time.perf_counter_ns()
        if event == "call":
            self.previous_line = None
            self.shell_rebuilds = 0
            self.status_reads = 0
            self.solver_submissions = 0
            self.maximum_candidate_pairs = 0
        if self.previous_line is not None:
            self.nanoseconds[self.previous_line] += tick - self.previous_tick
        if event == "line":
            self.previous_line = frame.f_lineno - self.first_line
            self.hits[self.previous_line] += 1
            source = self.source[self.previous_line].strip()
            self.shell_rebuilds += source.startswith("initialize_shell_bounds(")
            self.status_reads += source.startswith("status = qd_to_numpy(")
            # The first argument emits one event; a multiline call expression emits two.
            self.solver_submissions += source == "vertex_offsets,"
            if source.startswith("slots ="):
                self.maximum_candidate_pairs = max(self.maximum_candidate_pairs, frame.f_locals["n_contacts"])
        elif event == "return":
            self.substeps.append(
                SubstepCounts(
                    frame.f_locals["iteration"],
                    frame.f_locals["n_contacts"],
                    self.shell_rebuilds,
                    self.status_reads,
                    self.solver_submissions,
                    self.maximum_candidate_pairs,
                )
            )
            self.previous_line = None
        self.previous_tick = time.perf_counter_ns()
        return self.trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("volume", "shell"), default="volume")
    parser.add_argument("--mode", choices=("baseline", "kernels", "python", "lines", "nsys"), default="baseline")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads-per-vertex", type=int, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--line-search",
        dest="is_line_search_enabled",
        action="store_true",
        help="Enable local energy backtracking at extra cost.",
    )
    parser.add_argument("--tolerance", type=float, default=0.0, help="Zero runs a fixed iteration count.")
    parser.add_argument("--contact-distance", type=float, default=0.00001, help="Contact activation gap in metres.")
    parser.add_argument("--search-radius", type=float, default=0.002, help="Shell candidate search radius in metres.")
    parser.add_argument("--no-bending", dest="is_bending_enabled", action="store_false")
    parser.add_argument(
        "--contact-diagnostics",
        dest="is_contact_diagnostics_enabled",
        action="store_true",
        help="Check for penetration and report warnings at extra cost.",
    )
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
    started = time.perf_counter()
    gs.init(backend=gs.cuda, precision="32", logging_level=logging.WARNING)
    asset = Path(gs.__file__).parent / "assets" / "meshes" / "jelly" / "remeshed.obj"
    mesh = trimesh.load(asset, force="mesh", process=False)
    mesh.vertices[:, 2] += 0.015 - mesh.vertices[:, 2].min()
    vertices, elements = None, None
    if args.model == "volume":
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
            contact_d_hat=args.contact_distance,
            contact_search_radius=args.search_radius,
            is_contact_diagnostics_enabled=args.is_contact_diagnostics_enabled,
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
    if args.model == "volume":
        morph = gs.morphs.TetrahedralMesh(
            vertices=vertices,
            elements=elements,
        )
        material = gs.materials.VBD.Elastic(E=10000.0, nu=0.25, rho=360.0, friction_mu=0.4)
    else:
        morph = gs.morphs.MeshSet(
            files=(mesh,),
        )
        material = gs.materials.VBD.Cloth(
            E=100000.0,
            nu=0.25,
            rho=360.0,
            friction_mu=0.4,
            shear_stiffness=8.0,
            thickness=0.00025,
            bending_model="DiscreteShellBending" if args.is_bending_enabled else None,
            bending_rigidity=5e-8,
        )
    entity = scene.add_entity(
        morph=morph,
        material=material,
    )
    if args.threads_per_vertex is not None:
        scene.vbd_solver.config.n_vertex_threads = args.threads_per_vertex
    scene.build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(args.warmup):
        scene.step()
    torch.cuda.synchronize()
    setup_and_warmup_seconds = time.perf_counter() - started
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
        elif args.mode == "nsys":
            torch.cuda.profiler.start()
        try:
            for step in range(args.steps):
                tick = time.perf_counter()
                scene.step()
                torch.cuda.synchronize()
                timings.append(FrameTiming(repeat + 1, step + 1, (time.perf_counter() - tick) * 1000))
        finally:
            python_profiler.disable()
            sys.settrace(None)
            if args.mode == "nsys":
                torch.cuda.profiler.stop()
        print(
            f"Repeat {repeat + 1}: {np.mean([row.milliseconds for row in timings[-args.steps :]]):.3f} ms/frame",
            flush=True,
        )
    stem = f"{args.mode}_i{args.iterations}_tol{args.tolerance:g}_ls{'on' if args.is_line_search_enabled else 'off'}"
    stem += f"_gap{args.contact_distance:g}"
    stem = f"{args.model}_{stem}_radius{args.search_radius:g}_bend{int(args.is_bending_enabled)}"
    if args.mode == "python":
        python_profiler.dump_stats(args.output_dir / f"{stem}.pstats")
        with (args.output_dir / f"{stem}.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(python_profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(80)
            pstats.Stats(python_profiler, stream=stream).strip_dirs().sort_stats("tottime").print_stats(50)
    elif args.mode == "kernels":
        total_device_ms = qd.profiler.get_kernel_profiler_total_time() * 1000
        records = qd.lang.impl.get_runtime().prog.get_kernel_profiler_records()
        for name in sorted({record.name for record in records}):
            durations = [record.kernel_time for record in records if record.name == name]
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
    minimum_volume_ratio = None
    if args.model == "volume":
        rest_edges = vertices[elements[:, 1:]] - vertices[elements[:, :1]]
        final_edges = final_positions[elements[:, 1:]] - final_positions[elements[:, :1]]
        minimum_volume_ratio = float((np.linalg.det(final_edges) / np.linalg.det(rest_edges)).min())
    rest_faces = mesh.vertices[mesh.faces]
    final_faces = final_positions[mesh.faces]
    rest_areas = np.linalg.norm(
        np.cross(rest_faces[:, 1] - rest_faces[:, 0], rest_faces[:, 2] - rest_faces[:, 0]), axis=1
    )
    final_areas = np.linalg.norm(
        np.cross(final_faces[:, 1] - final_faces[:, 0], final_faces[:, 2] - final_faces[:, 0]), axis=1
    )
    report = ProfileReport(
        mode=args.mode,
        model=args.model,
        revision=revision,
        genesis_path=gs.__file__,
        platform=platform.platform(),
        python=sys.version,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        asset=asset.relative_to(Path(gs.__file__).parent / "assets").as_posix(),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        vertices=entity.n_vertices,
        tetrahedra=entity.n_elements if args.model == "volume" else 0,
        surface_triangles=len(mesh.faces),
        threads_per_vertex=scene.vbd_solver.config.n_vertex_threads,
        color_sizes=np.diff(scene.vbd_solver.colors.offsets).tolist(),
        frame_dt=scene.dt,
        substeps=scene.sim.substeps,
        iteration_limit=args.iterations,
        convergence_tolerance=args.tolerance,
        precision="32",
        material_model=material.model if isinstance(material, gs.materials.VBD.Elastic) else "plane_stress_stvk",
        youngs_modulus=entity.material.E,
        poisson_ratio=entity.material.nu,
        density=entity.material.rho,
        friction=entity.material.friction_mu,
        contact_distance=scene.vbd_options.contact_d_hat,
        contact_stiffness=scene.vbd_options.contact_stiffness,
        shell_contact_stiffness=scene.vbd_options.shell_contact_stiffness,
        contact_search_radius=scene.vbd_options.contact_search_radius,
        thickness=material.thickness if isinstance(material, gs.materials.VBD.Cloth) else None,
        bending_rigidity=material.bending_rigidity if isinstance(material, gs.materials.VBD.Cloth) else None,
        shear_stiffness=material.shear_stiffness if isinstance(material, gs.materials.VBD.Cloth) else None,
        is_bending_enabled=isinstance(material, gs.materials.VBD.Cloth) and args.is_bending_enabled,
        is_contact_diagnostics_enabled=scene.vbd_options.is_contact_diagnostics_enabled,
        is_line_search_enabled=scene.vbd_options.is_line_search_enabled,
        line_search_iterations=scene.vbd_options.line_search_iterations,
        setup_and_warmup_seconds=setup_and_warmup_seconds,
        warmup_frames=args.warmup,
        measured_frames_per_repeat=args.steps,
        repeats=args.repeats,
        mean_frame_ms=np.mean([row.milliseconds for row in timings]),
        median_frame_ms=np.median([row.milliseconds for row in timings]),
        p95_frame_ms=np.percentile([row.milliseconds for row in timings], 95),
        total_device_ms=total_device_ms,
        final_minimum_height=float(final_positions[:, 2].min()),
        final_minimum_volume_ratio=minimum_volume_ratio,
        final_minimum_area_ratio=float((final_areas / rest_areas).min()),
        timings=timings,
        kernels=kernels,
        lines=lines,
        substep_counts=line_timer.substeps,
    )
    report_path = args.output_dir / f"{stem}.json"
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
