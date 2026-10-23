"""Drop or load a jelly using vertex block descent (VBD), or finite elements with incremental potential contact."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import tetgen
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


@dataclass
class Benchmark:
    solver: str
    asset: str
    asset_sha256: str
    device: str
    device_name: str
    precision: str
    material_model: str
    vertices: int
    tetrahedra: int
    surface_triangles: int
    dt: float
    substeps: int
    scene_dt: float
    scene_substeps: int
    iterations: int | None
    contact_d_hat: float
    contact_stiffness: float | None
    friction_epsilon: float
    convergence_tolerance: float | None
    youngs_modulus: float
    poisson_ratio: float
    density: float
    friction: float
    is_plastic: bool
    yield_strain_percent: float
    hardening: float
    initialization_seconds: float
    simulated_seconds: float
    repeats: int
    mean_step_ms: float
    p95_step_ms: float
    wall_seconds: float
    minimum_height: float
    final_center: tuple[float, float, float]
    minimum_volume_ratio: float
    maximum_plastic_strain: float | None
    load_force: float | None
    peak_loaded_displacement: float | None
    residual_loaded_displacement: float | None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gpu", action="store_true")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("--solver", choices=("vbd", "ipc"), default="vbd")
    parser.add_argument("--scenario", choices=("drop", "load_unload"), default="drop")
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument(
        "--substeps", type=int, default=2, help="More substeps reduce integration damping at extra cost."
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--plastic", action="store_true")
    parser.add_argument("--yield-strain", type=float, default=10.0)
    parser.add_argument("--hardening", type=float, default=0.0)
    parser.add_argument("--youngs-modulus", type=float, help="Pa; defaults to 10000 for drop, 100000 for load_unload.")
    parser.add_argument("--poisson-ratio", type=float, default=0.25)
    parser.add_argument("--density", type=float, default=360.0)
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--contact-distance", type=float, default=0.0005)
    parser.add_argument("--tet-max-volume", type=float, default=1e-7)
    parser.add_argument(
        "--load-force", type=float, default=20.0, help="Total horizontal load in newtons before unloading."
    )
    parser.add_argument("--benchmark", action="store_true", help="Warm up 20 steps, then time five reset runs.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out/vbd"))
    args = parser.parse_args()
    if args.steps <= 0 or args.substeps <= 0 or args.tet_max_volume <= 0 or not np.isfinite(args.load_force):
        parser.error("--steps, --substeps and --tet-max-volume must be positive; --load-force must be finite")
    if args.solver == "ipc" and (not args.gpu or args.plastic or args.scenario != "drop"):
        parser.error("The FEM/IPC baseline uses --gpu, elastic material and --scenario drop.")
    youngs_modulus = args.youngs_modulus
    if youngs_modulus is None:
        youngs_modulus = 10000.0 if args.scenario == "drop" else 100000.0
    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision=args.precision)
    started = time.perf_counter()
    asset = Path(gs.__file__).parent / "assets" / "meshes" / "jelly" / "remeshed.obj"
    mesh = trimesh.load(asset, force="mesh", process=False)
    mesh.vertices[:, 2] += 0.015 - mesh.vertices[:, 2].min()
    tetrahedralizer = tetgen.TetGen(mesh.vertices, mesh.faces.astype(np.int32))
    vertices, elements, *_ = tetrahedralizer.tetrahedralize(switches=f"pq1.5/10Ya{args.tet_max_volume:.12f}Q")
    if not np.allclose(vertices[: len(mesh.vertices)], mesh.vertices):
        raise ValueError("Tetrahedralization must preserve the original surface mapping.")
    coupler = None
    material = gs.materials.VBD.Elastic(
        E=youngs_modulus,
        nu=args.poisson_ratio,
        rho=args.density,
        friction_mu=args.friction,
        is_volume_plasticity_enabled=args.plastic,
        volume_yield_strain_percent=args.yield_strain,
        volume_plastic_hardening=args.hardening,
    )
    if args.solver == "ipc":
        coupler = gs.options.IPCCouplerOptions(
            contact_d_hat=args.contact_distance,
            contact_eps_velocity=0.001,
        )
        material = gs.materials.FEM.Elastic(
            E=youngs_modulus,
            nu=args.poisson_ratio,
            rho=args.density,
            friction_mu=args.friction,
            model="stable_neohookean",
        )
    frame_dt, frame_substeps = 0.005, args.substeps
    scene_steps = frame_substeps if args.solver == "ipc" else 1
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=frame_dt / scene_steps,
            substeps=frame_substeps // scene_steps,
            gravity=(0, 0, -9.81),
        ),
        coupler_options=coupler,
        vbd_options=gs.options.VBDOptions(
            iterations=args.iterations,
            convergence_tolerance=args.tolerance,
            contact_d_hat=args.contact_distance,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.22, -0.28, 0.2),
            camera_lookat=(0.0, 0.0, 0.035),
        ),
        # profiling_options=gs.options.ProfilingOptions(
        #     show_FPS=False,
        # ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            vertices=vertices,
            elements=elements,
        ),
        material=material,
        surface=gs.surfaces.Default(color=(0.85, 0.3, 0.1, 1.0)),
    )
    if args.scenario == "load_unload":
        entity.set_vertex_constraints(np.flatnonzero(entity.init_positions[:, 2] < 0.016))
        loaded_indices = np.flatnonzero(entity.init_positions[:, 2] > entity.init_positions[:, 2].max() - 0.005)
    scene.build()
    if args.gpu:
        torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - started
    if args.benchmark:
        for _ in range(20):
            for _ in range(scene_steps):
                scene.step()
        scene.reset()
    durations = []
    peak_displacement = None
    repeats = 5 if args.benchmark else 1
    started = time.perf_counter()
    for repeat in range(repeats):
        if repeat:
            scene.reset()
        for step in range(args.steps):
            if args.scenario == "load_unload" and step == 0:
                entity.set_vertex_forces([args.load_force / len(loaded_indices), 0.0, 0.0], loaded_indices)
            if args.scenario == "load_unload" and step == args.steps // 2:
                positions = tensor_to_array(entity.get_state().pos)[0]
                peak_displacement = np.linalg.norm((positions - entity.init_positions)[loaded_indices].mean(axis=0))
                entity.clear_vertex_forces()
            tick = time.perf_counter()
            for _ in range(scene_steps):
                scene.step()
            if args.gpu:
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - tick)
        if args.benchmark:
            print(f"Run {repeat + 1}/{repeats}: mean {np.mean(durations[-args.steps :]) * 1000:.3f} ms", flush=True)
    elapsed = time.perf_counter() - started
    final_state = entity.get_state()
    final_positions = tensor_to_array(final_state.pos)[0]
    final_edges = final_positions[elements[:, 1:]] - final_positions[elements[:, :1]]
    rest_edges = vertices[elements[:, 1:]] - vertices[elements[:, :1]]
    volume_ratio = np.linalg.det(final_edges) / np.linalg.det(rest_edges)
    residual_displacement = None
    if args.scenario == "load_unload":
        residual_displacement = np.linalg.norm((final_positions - entity.init_positions)[loaded_indices].mean(axis=0))
    result = Benchmark(
        args.solver,
        asset.relative_to(Path(gs.__file__).parent / "assets").as_posix(),
        hashlib.sha256(asset.read_bytes()).hexdigest(),
        str(gs.device),
        torch.cuda.get_device_name() if args.gpu else "CPU",
        args.precision,
        "log_neohookean" if args.solver == "vbd" else "stable_neohookean",
        entity.n_vertices,
        entity.n_elements,
        len(mesh.faces),
        frame_dt,
        frame_substeps,
        scene.dt,
        scene.sim.substeps,
        args.iterations if args.solver == "vbd" else None,
        args.contact_distance,
        scene.vbd_options.contact_stiffness if args.solver == "vbd" else None,
        scene.vbd_options.friction_epsilon if args.solver == "vbd" else scene.sim.coupler_options.contact_eps_velocity,
        args.tolerance if args.solver == "vbd" else None,
        youngs_modulus,
        args.poisson_ratio,
        args.density,
        args.friction,
        args.plastic,
        args.yield_strain,
        args.hardening,
        initialization_seconds,
        args.steps * frame_dt,
        repeats,
        np.mean(durations) * 1000,
        np.percentile(durations, 95) * 1000,
        elapsed,
        float(final_positions[:, 2].min()),
        tuple(final_positions.mean(axis=0).tolist()),
        float(volume_ratio.min()),
        float(final_state.plastic_strain.max()) if args.solver == "vbd" else None,
        args.load_force if args.scenario == "load_unload" else None,
        float(peak_displacement) if peak_displacement is not None else None,
        float(residual_displacement) if residual_displacement is not None else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / f"{args.solver}_elastic_{args.scenario}.json"
    report.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(report)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
