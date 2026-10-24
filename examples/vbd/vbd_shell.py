"""Simulate the jelly surface or a folded sheet with vertex block descent (VBD)."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


@dataclass
class Benchmark:
    asset: str | None
    asset_sha256: str | None
    device: str
    device_name: str
    precision: str
    vertices: int
    triangles: int
    dt: float
    substeps: int
    iterations: int
    is_line_search_enabled: bool
    contact_d_hat: float
    contact_stiffness: float
    friction_epsilon: float
    convergence_tolerance: float
    thickness: float
    youngs_modulus: float
    poisson_ratio: float
    density: float
    shear_stiffness: float
    bending_rigidity: float
    friction: float
    is_plastic: bool
    yield_angle_deg: float
    hardening: float
    initialization_seconds: float
    simulated_seconds: float
    repeats: int
    mean_step_ms: float
    p95_step_ms: float
    wall_seconds: float
    minimum_height: float


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gpu", action="store_true", help="Run the full jelly mesh on the GPU.")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("--scenario", choices=("jelly", "fold"), default="jelly")
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--line-search",
        dest="is_line_search_enabled",
        action="store_true",
        help="Enable local energy backtracking at extra cost.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--plastic", action="store_true")
    parser.add_argument("--yield-angle", type=float, default=10.0)
    parser.add_argument("--hardening", type=float, default=0.0)
    parser.add_argument("--youngs-modulus", type=float, default=100000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.25)
    parser.add_argument("--density", type=float, default=360.0)
    parser.add_argument("--thickness", type=float, default=0.00025)
    parser.add_argument("--shear-stiffness", type=float, default=8.0)
    parser.add_argument("--bending-rigidity", type=float, default=5e-8)
    parser.add_argument("--no-bending", action="store_true")
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--contact-distance", type=float, default=0.00001)
    parser.add_argument("--benchmark", action="store_true", help="Warm up 20 steps, then time five reset runs.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out/vbd"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision=args.precision)
    started = time.perf_counter()
    asset = None
    if args.scenario == "jelly":
        asset = Path(gs.__file__).parent / "assets" / "meshes" / "jelly" / "remeshed.obj"
        mesh = trimesh.load(asset, force="mesh", process=False)
        mesh.vertices[:, 2] += 0.015 - mesh.vertices[:, 2].min()
    else:
        x, y = np.meshgrid(np.linspace(-0.04, 0.04, 17), np.linspace(-0.03, 0.03, 13))
        vertices = np.column_stack((x.flat, y.flat, np.full(x.size, 0.012)))
        cells = np.arange(12)[:, None] * 17 + np.arange(16)[None, :]
        cells = cells.reshape(cells.size)
        faces = np.concatenate(
            (np.column_stack((cells, cells + 1, cells + 17)), np.column_stack((cells + 1, cells + 18, cells + 17)))
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
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
            is_line_search_enabled=args.is_line_search_enabled,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.22, -0.28, 0.2),
            camera_lookat=(0.0, 0.0, 0.035),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    entity = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(mesh,),
        ),
        material=gs.materials.VBD.Cloth(
            E=args.youngs_modulus,
            nu=args.poisson_ratio,
            rho=args.density,
            friction_mu=args.friction,
            shear_stiffness=args.shear_stiffness,
            thickness=args.thickness,
            bending_model=None if args.no_bending else "DiscreteShellBending",
            bending_rigidity=args.bending_rigidity,
            is_bending_plasticity_enabled=args.plastic,
            bending_yield_angle_deg=args.yield_angle,
            bending_plastic_hardening=args.hardening,
        ),
        surface=gs.surfaces.Default(color=(0.7, 0.15, 0.3, 1.0)),
    )
    if args.scenario == "fold":
        fixed_indices = np.flatnonzero(entity.init_positions[:, 0] < -0.039)
        moving_indices = np.flatnonzero(entity.init_positions[:, 0] > 0.039)
        entity.set_vertex_constraints(fixed_indices)
        entity.set_vertex_constraints(moving_indices)
        moving_rest = entity.init_positions[moving_indices].copy()
    scene.build()
    if args.gpu:
        torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - started
    if args.benchmark:
        for _ in range(20):
            scene.step()
        scene.reset()
    durations = []
    minimum_height = np.inf
    repeats = 5 if args.benchmark else 1
    started = time.perf_counter()
    for repeat in range(repeats):
        if repeat:
            scene.reset()
        for step in range(args.steps):
            if args.scenario == "fold" and step < args.steps // 2:
                angle = np.pi * min(1.0, 2 * step / max(1, args.steps - 1))
                targets = moving_rest.copy()
                targets[:, 0] = 0.04 * np.cos(angle)
                targets[:, 2] = 0.012 + 0.04 * np.sin(angle) + 0.001 * angle / np.pi
                entity.set_vertex_constraints(moving_indices, targets)
            elif args.scenario == "fold" and step == args.steps // 2:
                entity.remove_vertex_constraints(moving_indices)
            tick = time.perf_counter()
            scene.step()
            if args.gpu:
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - tick)
        minimum_height = min(minimum_height, tensor_to_array(entity.get_state().pos)[..., 2].min())
        if args.benchmark:
            print(f"Run {repeat + 1}/{repeats}: mean {np.mean(durations[-args.steps :]) * 1000:.3f} ms", flush=True)
    elapsed = time.perf_counter() - started
    result = Benchmark(
        asset.relative_to(Path(gs.__file__).parent / "assets").as_posix() if asset is not None else None,
        hashlib.sha256(asset.read_bytes()).hexdigest() if asset is not None else None,
        str(gs.device),
        torch.cuda.get_device_name() if args.gpu else "CPU",
        args.precision,
        entity.n_vertices,
        entity.n_elements,
        scene.dt,
        scene.sim.substeps,
        args.iterations,
        args.is_line_search_enabled,
        args.contact_distance,
        scene.vbd_options.contact_stiffness,
        scene.vbd_options.friction_epsilon,
        args.tolerance,
        args.thickness,
        args.youngs_modulus,
        args.poisson_ratio,
        args.density,
        args.shear_stiffness,
        0.0 if args.no_bending else args.bending_rigidity,
        args.friction,
        args.plastic,
        args.yield_angle,
        args.hardening,
        initialization_seconds,
        args.steps * scene.dt,
        repeats,
        np.mean(durations) * 1000,
        np.percentile(durations, 95) * 1000,
        elapsed,
        float(minimum_height),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / f"vbd_shell_{args.scenario}.json"
    report.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(report)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
